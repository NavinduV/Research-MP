"""
FastAPI backend for the Microplastic Detection Pipeline Dashboard.

Exposes endpoints for:
  POST /api/detect       - Upload image(s) + run pipeline, returns full JSON report
  GET  /api/results/{id} - Retrieve a previously run result
  GET  /api/image/{id}   - Serve the annotated visualization
  GET  /api/mask/{id}    - Serve the mask overlay image
  GET  /api/health       - Service health / model loading status

Run with:
    uvicorn src.api.pipeline_api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Numpy-safe JSON helper
# ---------------------------------------------------------------------------

def _np_safe(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _np_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Microplastic Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent     # repo root
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job registry  { job_id: { status, result, error, created_at } }
_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Lazy model cache – models loaded once per process
# ---------------------------------------------------------------------------

_model_cache: dict = {}

def _get_models(
    yolo_path: str,
    maskrcnn_path: str,
    effnet_path: str,
    use_maskrcnn: bool,
    use_effnet: bool,
):
    """Load (or return cached) models."""
    import torch
    from ultralytics import YOLO
    from src.pipeline_inference import load_maskrcnn, load_effnet  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_key = f"yolo:{yolo_path}"
    if yolo_key not in _model_cache:
        _model_cache[yolo_key] = YOLO(yolo_path)

    maskrcnn_model = None
    if use_maskrcnn:
        mk_key = f"maskrcnn:{maskrcnn_path}"
        if mk_key not in _model_cache:
            _model_cache[mk_key] = load_maskrcnn(maskrcnn_path, device)
        maskrcnn_model = _model_cache[mk_key]

    effnet_model = None
    if use_effnet:
        eff_key = f"effnet:{effnet_path}"
        if eff_key not in _model_cache:
            _model_cache[eff_key] = load_effnet(effnet_path, device)
        effnet_model = _model_cache[eff_key]

    return device, _model_cache[yolo_key], maskrcnn_model, effnet_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_model_path(name: str) -> str:
    candidates = {
        "yolo": [
            "experiments/yolo/best.pt",
            "experiments/microplastic_yolo/weights/best.pt",
            "experiments/microplastic_yolo_max_accuracy/weights/best.pt",
            "experiments/yolo/weights/best.pt",
        ],
        "maskrcnn": [
            "experiments/maskrcnn/maskrcnn_crops_best.pth",
            "experiments/maskrcnn_crops_best.pth",
        ],
        "effnet": [
            "experiments/efficientnet/efficientnet_best.pth",
            "experiments/efficientnet_best.pth",
        ],
    }
    for rel in candidates.get(name, []):
        full = BASE_DIR / rel
        if full.exists():
            return str(full)
    return str(BASE_DIR / candidates[name][0])


def _build_summary(detections: list[dict], pixel_to_micron: float) -> dict:
    """Aggregate statistics from detection results."""
    from src.pipeline_inference import EFFNET_CLASS_NAMES

    if not detections:
        return {
            "total": 0,
            "counts": {"fiber": 0, "film": 0, "fragment": 0},
            "per_class": {},
            "overall": {},
        }

    counts = {cls: 0 for cls in EFFNET_CLASS_NAMES}
    per_class: dict[str, dict] = {}

    for det in detections:
        cls = det["final_class"]
        counts[cls] = counts.get(cls, 0) + 1

        si = det["size"]
        if cls not in per_class:
            per_class[cls] = {
                "lengths": [], "widths": [], "areas": [],
                "circularities": [], "aspect_ratios": [],
                "yolo_confidences": [], "effnet_confidences": [],
            }
        scale = pixel_to_micron
        per_class[cls]["lengths"].append(round(si["length_px"] * scale, 2))
        per_class[cls]["widths"].append(round(si["width_px"] * scale, 2))
        per_class[cls]["areas"].append(round(si["area_px"] * (scale ** 2), 2))
        per_class[cls]["circularities"].append(si["circularity"])
        per_class[cls]["aspect_ratios"].append(si["aspect_ratio"])
        per_class[cls]["yolo_confidences"].append(det["yolo_confidence"])
        per_class[cls]["effnet_confidences"].append(det["effnet_confidence"])

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {}
        arr = np.array(vals, dtype=float)
        return {
            "mean": round(float(arr.mean()), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(arr.std()), 3),
            "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
        }

    summary_per_class = {}
    for cls, data in per_class.items():
        summary_per_class[cls] = {
            "count": counts[cls],
            "length": _stats(data["lengths"]),
            "width": _stats(data["widths"]),
            "area": _stats(data["areas"]),
            "circularity": _stats(data["circularities"]),
            "aspect_ratio": _stats(data["aspect_ratios"]),
            "yolo_confidence": _stats(data["yolo_confidences"]),
            "effnet_confidence": _stats(data["effnet_confidences"]),
        }

    all_lengths = [si["length_px"] * pixel_to_micron for d in detections for si in [d["size"]]]
    all_areas = [si["area_px"] * (pixel_to_micron ** 2) for d in detections for si in [d["size"]]]
    all_circ = [d["size"]["circularity"] for d in detections]
    all_ar = [d["size"]["aspect_ratio"] for d in detections]

    overall = {
        "length": _stats(all_lengths),
        "area": _stats(all_areas),
        "circularity": _stats(all_circ),
        "aspect_ratio": _stats(all_ar),
    }

    # Distribution buckets for histogram data (length)
    unit = "um" if pixel_to_micron != 1.0 else "px"
    hist, bin_edges = np.histogram(all_lengths, bins=10)
    length_histogram = {
        "unit": unit,
        "counts": hist.tolist(),
        "bin_edges": [round(float(e), 2) for e in bin_edges],
    }

    return {
        "total": len(detections),
        "counts": counts,
        "per_class": summary_per_class,
        "overall": overall,
        "length_histogram": length_histogram,
        "unit": unit,
        "pixel_to_micron": pixel_to_micron,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Quick health check."""
    import torch
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "jobs_in_memory": len(_jobs),
        "default_models": {
            "yolo": _default_model_path("yolo"),
            "maskrcnn": _default_model_path("maskrcnn"),
            "effnet": _default_model_path("effnet"),
        },
    }


@app.post("/api/detect")
async def detect(
    files: list[UploadFile] = File(...),
    yolo_path: str = Form(default=""),
    maskrcnn_path: str = Form(default=""),
    effnet_path: str = Form(default=""),
    yolo_conf: float = Form(default=0.1),
    mask_threshold: float = Form(default=0.5),
    pixel_to_micron: float = Form(default=1.0),
    crop_padding: int = Form(default=30),
    nms_iou: float = Form(default=0.3),
    use_maskrcnn: bool = Form(default=True),
    use_effnet: bool = Form(default=True),
):
    """
    Run the full pipeline on one or more uploaded images.

    Returns a JSON job report with detections, statistics, and image IDs
    that can be used to fetch result images via /api/image/{job_id}/{index}.
    """
    from src.pipeline_inference import (
        EFFNET_CLASS_NAMES,
        get_effnet_transform,
        get_maskrcnn_transform,
        run_yolo_detection,
        class_agnostic_nms,
        crop_detection,
        classify_crop,
        segment_crop,
        generate_ellipse_mask,
        calculate_microplastic_size,
        _create_visualization,
        COLORS,
    )
    import torch

    yolo_path = yolo_path or _default_model_path("yolo")
    maskrcnn_path = maskrcnn_path or _default_model_path("maskrcnn")
    effnet_path = effnet_path or _default_model_path("effnet")

    job_id = str(uuid.uuid4())
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    img_upload_dir = UPLOAD_DIR / job_id
    img_upload_dir.mkdir(parents=True, exist_ok=True)

    _jobs[job_id] = {"status": "running", "created_at": time.time()}

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load models ----
        from ultralytics import YOLO
        from src.pipeline_inference import load_maskrcnn, load_effnet

        yolo_model = YOLO(yolo_path)
        maskrcnn_model = load_maskrcnn(maskrcnn_path, device) if use_maskrcnn else None
        effnet_model = load_effnet(effnet_path, device) if use_effnet else None
        if use_effnet and effnet_model is None:
            use_effnet = False

        effnet_transform = get_effnet_transform() if use_effnet else None
        maskrcnn_transform = get_maskrcnn_transform() if use_maskrcnn else None

        image_results = []

        for file_idx, upload in enumerate(files):
            # Save uploaded file
            ext = Path(upload.filename).suffix or ".png"
            img_filename = f"input_{file_idx}{ext}"
            img_path = img_upload_dir / img_filename
            with open(img_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            image = cv2.imread(str(img_path))
            if image is None:
                image_results.append({
                    "filename": upload.filename,
                    "error": "Could not decode image",
                })
                continue

            h, w = image.shape[:2]

            # Stage 1: YOLO
            detections_raw = run_yolo_detection(yolo_model, str(img_path), yolo_conf)

            # Stage 2: Class-agnostic NMS
            nms_input = [
                [d['box'][0], d['box'][1], d['box'][2], d['box'][3],
                 d['class_id'], d['confidence']]
                for d in detections_raw
            ]
            nms_kept = class_agnostic_nms(nms_input, iou_threshold=nms_iou)
            detections = []
            for bx in nms_kept:
                cid = int(bx[4])
                detections.append({
                    'box': [int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])],
                    'class_id': cid,
                    'confidence': float(bx[5]),
                    'class_name': EFFNET_CLASS_NAMES[cid],
                })

            # Stage 3 & 4: EfficientNet classify + Mask R-CNN mask
            mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            results_list = []

            for i, det in enumerate(detections):
                box = det["box"]
                yolo_class_id = det["class_id"]
                yolo_class_name = det["class_name"]
                yolo_conf_score = det["confidence"]

                crop, crop_box, rel_box = crop_detection(image, box, padding=crop_padding)

                if use_effnet and effnet_model is not None:
                    effnet_class_id, effnet_class_name, effnet_conf, effnet_probs = \
                        classify_crop(effnet_model, crop, device, effnet_transform)
                    final_class_name = effnet_class_name
                    final_class_id = effnet_class_id
                    classification_source = "effnet"
                else:
                    final_class_name = yolo_class_name
                    final_class_id = yolo_class_id
                    effnet_conf = 0.0
                    effnet_probs = None
                    classification_source = "yolo"

                if use_maskrcnn and maskrcnn_model is not None:
                    mask_crop, mask_conf = segment_crop(
                        maskrcnn_model, crop, final_class_id, device,
                        maskrcnn_transform, mask_threshold,
                    )
                    x1_pad, y1_pad, x2_pad, y2_pad = crop_box
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    full_mask[y1_pad:y2_pad, x1_pad:x2_pad] = mask_crop
                    segmentation_source = "maskrcnn"
                else:
                    full_mask = generate_ellipse_mask(box, image.shape)
                    mask_conf = 0.5
                    segmentation_source = "ellipse"

                size_info = calculate_microplastic_size(full_mask, pixel_to_micron)

                color = COLORS.get(final_class_name, (255, 255, 255))
                mask_overlay[full_mask == 1] = color

                result_item = {
                    "id": i + 1,
                    "box": box,
                    "yolo_class": yolo_class_name,
                    "yolo_confidence": round(yolo_conf_score, 4),
                    "final_class": final_class_name,
                    "final_class_id": final_class_id,
                    "classification_source": classification_source,
                    "effnet_confidence": round(effnet_conf, 4),
                    "effnet_probabilities": (
                        {EFFNET_CLASS_NAMES[j]: round(float(p), 4)
                         for j, p in enumerate(effnet_probs)}
                        if effnet_probs is not None else None
                    ),
                    "mask_confidence": round(mask_conf, 4),
                    "segmentation_source": segmentation_source,
                    "size": size_info,
                }
                results_list.append(result_item)

            # Save visualization
            vis = _create_visualization(image, results_list, mask_overlay, pixel_to_micron)
            vis_path = job_dir / f"vis_{file_idx}.jpg"
            cv2.imwrite(str(vis_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

            mask_path = job_dir / f"mask_{file_idx}.jpg"
            cv2.imwrite(str(mask_path), mask_overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Save original (copy)
            orig_path = job_dir / f"orig_{file_idx}{ext}"
            shutil.copy(str(img_path), str(orig_path))

            summary = _build_summary(results_list, pixel_to_micron)

            image_results.append({
                "filename": upload.filename,
                "image_size": {"width": w, "height": h},
                "file_index": file_idx,
                "detections": results_list,
                "summary": summary,
            })

        full_report = {
            "job_id": job_id,
            "created_at": time.time(),
            "config": {
                "yolo_conf": yolo_conf,
                "mask_threshold": mask_threshold,
                "pixel_to_micron": pixel_to_micron,
                "crop_padding": crop_padding,
                "nms_iou": nms_iou,
                "use_maskrcnn": use_maskrcnn,
                "use_effnet": use_effnet,
            },
            "images": image_results,
        }

        # Persist report
        report_path = job_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2, default=str)

        _jobs[job_id] = {"status": "done", "created_at": full_report["created_at"]}
        return JSONResponse(_np_safe(full_report))

    except Exception as exc:
        import traceback
        traceback.print_exc()   # prints full stack trace to uvicorn terminal
        _jobs[job_id] = {"status": "error", "error": str(exc)}
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/results/{job_id}")
async def get_result(job_id: str):
    """Return the stored JSON report for a completed job."""
    report_path = RESULTS_DIR / job_id / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    with open(report_path) as f:
        return JSONResponse(json.load(f))


@app.get("/api/image/{job_id}/{file_index}")
async def get_visualization(job_id: str, file_index: int):
    """Return the annotated visualization image for a specific result."""
    vis_path = RESULTS_DIR / job_id / f"vis_{file_index}.jpg"
    if not vis_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    def _iter():
        with open(vis_path, "rb") as f:
            yield from f

    return StreamingResponse(_iter(), media_type="image/jpeg")


@app.get("/api/mask/{job_id}/{file_index}")
async def get_mask(job_id: str, file_index: int):
    """Return the mask overlay image for a specific result."""
    mask_path = RESULTS_DIR / job_id / f"mask_{file_index}.jpg"
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Mask not found")

    def _iter():
        with open(mask_path, "rb") as f:
            yield from f

    return StreamingResponse(_iter(), media_type="image/jpeg")


@app.get("/api/original/{job_id}/{file_index}")
async def get_original(job_id: str, file_index: int):
    """Return the original uploaded image."""
    job_dir = RESULTS_DIR / job_id
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
        p = job_dir / f"orig_{file_index}{ext}"
        if p.exists():
            def _iter(path=p):
                with open(path, "rb") as f:
                    yield from f
            mt = "image/png" if ext in (".png",) else "image/jpeg"
            return StreamingResponse(_iter(), media_type=mt)
    raise HTTPException(status_code=404, detail="Original not found")


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs in memory."""
    return {"jobs": [{"job_id": k, **v} for k, v in _jobs.items()]}
