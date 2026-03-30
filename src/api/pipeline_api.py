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

# Allow large microscopy image uploads (default max_part_size is 1 MB).
# Monkey-patch Starlette's Request.form() to raise the per-part limit to 500 MB
# so large microscopy images / batches are accepted.

def _patch_multipart_limit():
    from starlette.requests import Request
    import functools

    _orig = Request.form

    @functools.wraps(_orig)
    def _form_with_large_limit(self, *, max_files=1000, max_fields=1000,
                                max_part_size=500 * 1024 * 1024):
        return _orig(self, max_files=max_files, max_fields=max_fields,
                     max_part_size=max_part_size)

    Request.form = _form_with_large_limit

_patch_multipart_limit()

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
    from src.pipeline.pipeline_inference import load_maskrcnn, load_effnet  # noqa: E402

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

def _default_model_path(name: str, mode: str = "macro") -> str:
    """Return the best available model path for the given mode (macro/micro)."""
    _CANDIDATES = {
        "macro": {
            "yolo": [
                "experiments/macro/yolo/best.pt",
                "experiments/macro/microplastic_yolo/weights/best.pt",
                "experiments/yolo/best.pt",
                "experiments/microplastic_yolo/weights/best.pt",
                "experiments/microplastic_yolo_max_accuracy/weights/best.pt",
                "experiments/yolo/weights/best.pt",
            ],
            "maskrcnn": [
                "experiments/macro/maskrcnn/maskrcnn_crops_best.pth",
                "experiments/maskrcnn/maskrcnn_crops_best.pth",
                "experiments/maskrcnn_crops_best.pth",
            ],
            "effnet": [
                "experiments/macro/efficientnet/efficientnet_best.pth",
                "experiments/efficientnet/efficientnet_best.pth",
                "experiments/efficientnet_best.pth",
            ],
        },
        "micro": {
            "yolo": [
                "experiments/micro/yolo/best.pt",
            ],
            "maskrcnn": [
                "experiments/micro/maskrcnn/maskrcnn_crops_best.pth",
            ],
            "effnet": [
                "experiments/micro/efficientnet/efficientnet_best.pth",
            ],
        },
    }
    candidates = _CANDIDATES.get(mode, _CANDIDATES["macro"])
    for rel in candidates.get(name, []):
        full = BASE_DIR / rel
        if full.exists():
            return str(full)
    return str(BASE_DIR / candidates[name][0])


def _build_summary(detections: list[dict], pixel_to_micron: float) -> dict:
    """Aggregate statistics from detection results."""
    from src.pipeline.pipeline_inference import EFFNET_CLASS_NAMES

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
            "macro": {
                "yolo": _default_model_path("yolo", "macro"),
                "maskrcnn": _default_model_path("maskrcnn", "macro"),
                "effnet": _default_model_path("effnet", "macro"),
            },
            "micro": {
                "yolo": _default_model_path("yolo", "micro"),
                "maskrcnn": _default_model_path("maskrcnn", "micro"),
                "effnet": _default_model_path("effnet", "micro"),
            },
        },
    }


@app.post("/api/detect")
async def detect(
    files: list[UploadFile] = File(...),
    pipeline_mode: str = Form(default="macro"),
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
    from src.pipeline.pipeline_inference import (
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
        load_per_type_maskrcnn,
        NUM_CLASSES_MASKRCNN,
    )
    from src.pipeline.filter_mask import detect_filter_circle_from_array
    import torch

    # Normalise mode
    pipeline_mode = pipeline_mode.strip().lower()
    if pipeline_mode not in ("macro", "micro"):
        pipeline_mode = "macro"

    # Enforce toggle dependency: Mask R-CNN requires EfficientNet
    if not use_effnet:
        use_maskrcnn = False

    # Log effective pipeline variant
    if use_effnet and use_maskrcnn:
        _variant = "Full Pipeline (YOLO + EfficientNet + Mask R-CNN)"
    elif use_effnet:
        _variant = "YOLO + EfficientNet (no Mask R-CNN)"
    else:
        _variant = "YOLO Only"
    print(f"[Pipeline] Mode={pipeline_mode}, Variant={_variant}")

    # Choose micro-specific per-type loader when in micro mode
    load_per_type_fn = load_per_type_maskrcnn
    if pipeline_mode == "micro":
        from src.pipeline.pipeline_inference_micro import (
            load_per_type_maskrcnn_micro,
            run_yolo_detection as run_yolo_detection_micro,
        )
        load_per_type_fn = load_per_type_maskrcnn_micro

    yolo_path = yolo_path or _default_model_path("yolo", pipeline_mode)
    maskrcnn_path = maskrcnn_path or _default_model_path("maskrcnn", pipeline_mode)
    effnet_path = effnet_path or _default_model_path("effnet", pipeline_mode)

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
        from src.pipeline.pipeline_inference import load_maskrcnn, load_effnet

        yolo_key = f"yolo:{yolo_path}"
        if yolo_key not in _model_cache:
            _model_cache[yolo_key] = YOLO(yolo_path)
            print(f"[YOLO] Loaded from: {yolo_path}")
        yolo_model = _model_cache[yolo_key]

        # Load per-type Mask R-CNN models (preferred)
        per_type_models: dict = {}
        fallback_maskrcnn = None
        if use_maskrcnn:
            per_type_cache_key = f"maskrcnn_per_type:{pipeline_mode}"
            if per_type_cache_key not in _model_cache:
                _model_cache[per_type_cache_key] = load_per_type_fn(device)
            per_type_models = _model_cache[per_type_cache_key]

            # Fallback for types without per-type model
            missing = [t for t in EFFNET_CLASS_NAMES if t not in per_type_models]
            if missing and maskrcnn_path and Path(maskrcnn_path).exists():
                mk_key = f"maskrcnn:{maskrcnn_path}"
                if mk_key not in _model_cache:
                    _model_cache[mk_key] = load_maskrcnn(maskrcnn_path, device,
                                                         num_classes=NUM_CLASSES_MASKRCNN)
                fallback_maskrcnn = _model_cache[mk_key]

        effnet_model = None
        if use_effnet:
            eff_key = f"effnet:{effnet_path}"
            if eff_key not in _model_cache:
                _model_cache[eff_key] = load_effnet(effnet_path, device)
            effnet_model = _model_cache[eff_key]
            if effnet_model is None:
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

            # Detect filter paper circle (multi-strategy)
            filter_circle = None
            full_coverage = False
            try:
                fc_center, fc_radius, _fc_mask, fc_method, full_coverage = \
                    detect_filter_circle_from_array(image)
                if not full_coverage:
                    filter_circle = (fc_center, fc_radius)
            except Exception:
                filter_circle = None

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

            # Stage 2b: Filter detections to filter-paper region only
            if filter_circle is not None and not full_coverage:
                fc_cx, fc_cy = filter_circle[0]
                fc_r = filter_circle[1]
                detections = [
                    d for d in detections
                    if np.sqrt(((d['box'][0]+d['box'][2])/2 - fc_cx)**2 +
                               ((d['box'][1]+d['box'][3])/2 - fc_cy)**2) <= fc_r
                ]

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
                    # Without EfficientNet, label all detections as generic "microplastic"
                    # because YOLO's type classification is not reliable on its own
                    final_class_name = "microplastic"
                    final_class_id = -1
                    effnet_conf = 0.0
                    effnet_probs = None
                    classification_source = "yolo"

                if use_maskrcnn:
                    chosen_model = per_type_models.get(final_class_name, fallback_maskrcnn)
                    if chosen_model is not None:
                        mask_crop, mask_conf = segment_crop(
                            chosen_model, crop, final_class_id, device,
                            maskrcnn_transform, mask_threshold,
                        )
                        x1_pad, y1_pad, x2_pad, y2_pad = crop_box
                        full_mask = np.zeros((h, w), dtype=np.uint8)
                        full_mask[y1_pad:y2_pad, x1_pad:x2_pad] = mask_crop
                        model_name = final_class_name if final_class_name in per_type_models else 'generic'
                        segmentation_source = f"maskrcnn_{model_name}"
                    else:
                        full_mask = generate_ellipse_mask(box, image.shape)
                        mask_conf = 0.5
                        segmentation_source = "ellipse"
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
            vis = _create_visualization(image, results_list, mask_overlay, pixel_to_micron,
                                        filter_circle=filter_circle,
                                        use_maskrcnn=use_maskrcnn)
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
                "filter_circle": {
                    "center": list(filter_circle[0]),
                    "radius": int(filter_circle[1]),
                } if filter_circle else None,
                "file_index": file_idx,
                "detections": results_list,
                "summary": summary,
            })

        full_report = {
            "job_id": job_id,
            "pipeline_mode": pipeline_mode,
            "created_at": time.time(),
            "config": {
                "pipeline_mode": pipeline_mode,
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
    """List all jobs — merges in-memory state with persisted results on disk."""
    all_jobs: dict[str, dict] = {}

    # 1. Scan results/ directory for persisted reports
    if RESULTS_DIR.exists():
        for job_dir in RESULTS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            report_path = job_dir / "report.json"
            if report_path.exists():
                try:
                    with open(report_path) as f:
                        report = json.load(f)
                    jid = report.get("job_id", job_dir.name)
                    all_jobs[jid] = {
                        "status": "done",
                        "created_at": report.get("created_at", 0),
                        "pipeline_mode": report.get("pipeline_mode",
                                                     report.get("config", {}).get("pipeline_mode", "macro")),
                        "total_detections": sum(
                            im.get("summary", {}).get("total", 0)
                            for im in report.get("images", [])
                        ),
                        "image_count": len(report.get("images", [])),
                    }
                except Exception:
                    pass

    # 2. Overlay in-memory state (may have running / error jobs not yet persisted)
    for k, v in _jobs.items():
        if k not in all_jobs or v.get("status") in ("running", "error"):
            all_jobs[k] = v

    return {"jobs": [{"job_id": k, **v} for k, v in all_jobs.items()]}


# ---------------------------------------------------------------------------
# Stitching – in-memory session cache
# ---------------------------------------------------------------------------

_stitch_cache: dict[str, dict] = {}

# Temp directory for uploaded stitch images
_STITCH_UPLOAD_DIR = BASE_DIR / "uploads" / "_stitch_temp"


@app.post("/api/stitch/upload-and-analyze")
async def stitch_upload_and_analyze(files: list[UploadFile] = File(...)):
    """
    Accept uploaded image files, save to a temp directory on disk,
    run brightness grouping, and return the analysis with server-side paths.
    This enables the "Stitch These" flow from the upload page.
    """
    from src.img_preprocess.macro_stitch_pipeline import group_images_by_brightness

    # Create a unique temp sub-directory
    session = str(uuid.uuid4())[:8]
    temp_dir = _STITCH_UPLOAD_DIR / session
    temp_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for upload in files:
        ext = Path(upload.filename).suffix or ".png"
        safe_name = upload.filename.replace(" ", "_")
        dest = temp_dir / safe_name
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_paths.append(str(dest.resolve()))

    # Run brightness analysis
    groups = group_images_by_brightness(str(temp_dir))
    if not groups:
        # If grouping fails, create a single group with all images
        groups_result = {"0": [
            {"path": p.replace("\\", "/"), "filename": Path(p).name, "brightness": 0}
            for p in saved_paths
        ]}
    else:
        groups_result = {}
        for brightness_level, images in sorted(groups.items()):
            key = str(int(brightness_level))
            groups_result[key] = [
                {
                    "path": img["path"].replace("\\", "/"),
                    "filename": img["filename"],
                    "brightness": round(img["brightness"], 1),
                }
                for img in images
            ]

    return {
        "groups": groups_result,
        "folder": str(temp_dir.resolve()).replace("\\", "/"),
        "uploaded_count": len(saved_paths),
    }


@app.post("/api/stitch/analyze")
async def stitch_analyze(folder_path: str = Form(...)):
    """Analyze a folder of images and return brightness groups for selection."""
    from src.img_preprocess.macro_stitch_pipeline import group_images_by_brightness

    folder = folder_path.strip()
    if not Path(folder).is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

    groups = group_images_by_brightness(folder)
    if not groups:
        raise HTTPException(status_code=404, detail="No images found in folder")

    # Serialize brightness groups for the frontend
    result = {}
    for brightness_level, images in sorted(groups.items()):
        key = str(int(brightness_level))
        result[key] = [
            {
                "path": img["path"].replace("\\", "/"),
                "filename": img["filename"],
                "brightness": round(img["brightness"], 1),
            }
            for img in images
        ]
    return {"groups": result, "folder": folder}


@app.get("/api/stitch/thumbnail")
async def stitch_thumbnail(path: str):
    """Serve a thumbnail preview of an image on disk."""
    full = Path(path)
    if not full.exists():
        raise HTTPException(status_code=404, detail="File not found")
    img = cv2.imread(str(full))
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    # Resize to thumbnail (max 240px)
    h, w = img.shape[:2]
    max_thumb = 240
    if max(h, w) > max_thumb:
        sc = max_thumb / max(h, w)
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.post("/api/stitch/run")
async def stitch_run(
    folder_path: str = Form(...),
    selected_images: str = Form(...),
    advanced_mode: bool = Form(False),
    output_name: str = Form("stitched_output.png"),
    max_dim: int = Form(8192),
    upscale: float = Form(1.0),
):
    """Run the stitching pipeline on selected images."""
    from src.img_preprocess.macro_stitch_pipeline import (
        stitch_images,
        stitch_images_advanced,
        prepare_for_yolo,
        upscale_image,
    )

    image_paths = [p.strip() for p in selected_images.split("|||") if p.strip()]
    if len(image_paths) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 images")

    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)
    if len(images) < 2:
        raise HTTPException(status_code=400, detail=f"Could not read enough images ({len(images)} loaded)")

    try:
        if advanced_mode and len(images) >= 3:
            pano = stitch_images_advanced(images, max_stitch_dim=10000)
        else:
            pano = stitch_images(images, max_stitch_dim=8192)

        if upscale > 1.0:
            pano = upscale_image(pano, scale=upscale)

        # Save to datasets/stitched/
        out_dir = BASE_DIR / "datasets" / "stitched"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / output_name)
        meta = prepare_for_yolo(pano, out_path, max_dim=max_dim)

        session_id = str(uuid.uuid4())[:8]
        abs_path = str(Path(out_path).resolve())
        _stitch_cache[session_id] = {
            "image_path": abs_path,
            "meta": meta,
            "num_images": len(images),
            "advanced": advanced_mode and len(images) >= 3,
        }

        return {
            "session_id": session_id,
            "meta": _np_safe(meta),
            "num_images": len(images),
            "advanced": advanced_mode and len(images) >= 3,
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/stitch/preview/{session_id}")
async def stitch_preview(session_id: str):
    """Serve the stitched result image."""
    if session_id not in _stitch_cache:
        raise HTTPException(status_code=404, detail="Session expired")
    img_path = _stitch_cache[session_id]["image_path"]
    if not Path(img_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    img = cv2.imread(img_path)
    if img is None:
        raise HTTPException(status_code=500, detail="Cannot read image")

    # Send a reasonably-sized preview (max 1600px)
    h, w = img.shape[:2]
    max_dim_preview = 1600
    if max(h, w) > max_dim_preview:
        sc = max_dim_preview / max(h, w)
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.post("/api/stitch/enhance/{session_id}")
async def stitch_enhance(
    session_id: str,
    sharpen: float = Form(0),
    denoise: int = Form(0),
    contrast: float = Form(0),
    brightness: int = Form(0),
    auto_wb: bool = Form(False),
):
    """Apply enhancements to the stitched image and save."""
    from src.img_preprocess.macro_stitch_pipeline import enhance_image

    if session_id not in _stitch_cache:
        raise HTTPException(status_code=404, detail="Session expired")

    img_path = _stitch_cache[session_id]["image_path"]
    img = cv2.imread(img_path)
    if img is None:
        raise HTTPException(status_code=500, detail="Cannot read image")

    if sharpen > 0 or denoise > 0 or contrast > 0 or brightness != 0 or auto_wb:
        img = enhance_image(
            img, sharpen=sharpen, denoise=denoise,
            contrast=contrast, brightness=brightness, auto_wb=auto_wb,
        )
        cv2.imwrite(img_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    return {"success": True, "path": img_path}


@app.get("/api/stitch/enhance-preview/{session_id}")
async def stitch_enhance_preview(
    session_id: str,
    sharpen: float = 0,
    denoise: int = 0,
    contrast: float = 0,
    brightness: int = 0,
    auto_wb: int = 0,
):
    """Return a live-preview JPEG with the requested enhancement settings."""
    from src.img_preprocess.macro_stitch_pipeline import enhance_image

    if session_id not in _stitch_cache:
        raise HTTPException(status_code=404, detail="Session expired")

    img_path = _stitch_cache[session_id]["image_path"]
    img = cv2.imread(img_path)
    if img is None:
        raise HTTPException(status_code=500, detail="Cannot read image")

    h, w = img.shape[:2]
    max_prev = 1200
    if max(h, w) > max_prev:
        sc = max_prev / max(h, w)
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)

    if sharpen > 0 or denoise > 0 or contrast > 0 or brightness != 0 or auto_wb:
        img = enhance_image(
            img, sharpen=sharpen, denoise=denoise,
            contrast=contrast, brightness=brightness,
            auto_wb=bool(auto_wb),
        )

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(
        io.BytesIO(buf.tobytes()), media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/api/stitch/send-to-pipeline/{session_id}")
async def stitch_send_to_pipeline(session_id: str):
    """
    Copy the stitched image into the uploads folder and return its path
    so the frontend can pass it through the normal /api/detect endpoint.
    Returns a blob URL-compatible file.
    """
    if session_id not in _stitch_cache:
        raise HTTPException(status_code=404, detail="Session expired")
    img_path = _stitch_cache[session_id]["image_path"]
    if not Path(img_path).exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Read the full-res file and stream it back so the frontend can POST it to /api/detect
    def _iter():
        with open(img_path, "rb") as f:
            yield from f

    ext = Path(img_path).suffix.lower()
    mt = "image/png" if ext == ".png" else "image/jpeg"
    fname = Path(img_path).name
    return StreamingResponse(
        _iter(), media_type=mt,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.delete("/api/stitch/{session_id}")
async def stitch_delete(session_id: str):
    """Delete stitched image and clear session."""
    if session_id in _stitch_cache:
        img_path = _stitch_cache[session_id]["image_path"]
        try:
            if Path(img_path).exists():
                os.remove(img_path)
        except Exception:
            pass
        del _stitch_cache[session_id]
    return {"deleted": True}
