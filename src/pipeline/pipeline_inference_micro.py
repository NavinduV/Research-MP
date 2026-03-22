"""
YOLO + EfficientNet + Per-Type Mask R-CNN Pipeline for Microplastic Detection,
Classification & Segmentation — MICRO IMAGE VARIANT.

This module mirrors the macro pipeline (pipeline_inference.py) but uses the
micro-specific models stored under ``experiments/micro/``.

Model layout expected under experiments/micro/:
    yolo/best.pt
    efficientnet/efficientnet_best.pth
    maskrcnn_fiber/maskrcnn_best.pth
    maskrcnn_film/maskrcnn_best.pth
    maskrcnn_fragment/maskrcnn_best.pth

Usage:
    python src/pipeline/pipeline_inference_micro.py --image path/to/image.png
"""

import argparse
import json
import cv2
import numpy as np
import torch

try:
    from src.pipeline.filter_mask import detect_filter_circle_from_array
except ImportError:
    from filter_mask import detect_filter_circle_from_array
from pathlib import Path
from ultralytics import YOLO
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms
import timm

# ---------------------------------------------------------------------------
# Import shared utilities from the macro pipeline to avoid duplication
# ---------------------------------------------------------------------------
try:
    from src.pipeline.pipeline_inference import (
        class_agnostic_nms,
        get_maskrcnn_model,
        load_maskrcnn,
        load_effnet,
        get_effnet_transform,
        get_maskrcnn_transform,
        crop_detection,
        classify_crop,
        segment_crop,
        generate_ellipse_mask,
        calculate_microplastic_size,
        _create_visualization,
        _print_size_summary,
    )
except ImportError:
    from pipeline_inference import (
        class_agnostic_nms,
        get_maskrcnn_model,
        load_maskrcnn,
        load_effnet,
        get_effnet_transform,
        get_maskrcnn_transform,
        crop_detection,
        classify_crop,
        segment_crop,
        generate_ellipse_mask,
        calculate_microplastic_size,
        _create_visualization,
        _print_size_summary,
    )

# ============================================================================
# Configuration — Micro-specific paths
# ============================================================================

NUM_CLASSES_BINARY = 2    # per-type models: background + 1 type
NUM_CLASSES_MASKRCNN = 4  # legacy single model: background + fiber + film + fragment
NUM_CLASSES_EFFNET = 3    # fiber, film, fragment

CLASS_NAMES = ['background', 'fiber', 'film', 'fragment']
EFFNET_CLASS_NAMES = ['fiber', 'film', 'fragment']

YOLO_TO_MASKRCNN_CLASS = {0: 1, 1: 2, 2: 3}
MASKRCNN_TO_CLASS_NAME = {1: 'fiber', 2: 'film', 3: 'fragment'}

# Micro-specific per-type model paths
MICRO_PER_TYPE_MODEL_CANDIDATES = {
    'fiber': [
        'experiments/micro/maskrcnn_fiber/maskrcnn_best.pth',
        'experiments/micro/maskrcnn_fiber/maskrcnn_fiber_best.pth',
    ],
    'film': [
        'experiments/micro/maskrcnn_film/maskrcnn_best.pth',
        'experiments/micro/maskrcnn_film/maskrcnn_film_best.pth',
    ],
    'fragment': [
        'experiments/micro/maskrcnn_fragment/maskrcnn_best.pth',
        'experiments/micro/maskrcnn_fragment/maskrcnn_fragment_best.pth',
    ],
}

# Default micro model paths
MICRO_YOLO_PATH = 'experiments/micro/yolo/best.pt'
MICRO_EFFNET_PATH = 'experiments/micro/efficientnet/efficientnet_best.pth'
MICRO_MASKRCNN_FALLBACK_PATH = 'experiments/micro/maskrcnn/maskrcnn_crops_best.pth'

# Colors for visualization (BGR format for OpenCV)
COLORS = {
    'fiber':    (0, 0, 255),     # red
    'film':     (0, 255, 255),   # yellow
    'fragment': (0, 255, 0),     # green
}


# ============================================================================
# YOLO Detection (mirrors macro, but can load micro model)
# ============================================================================

def run_yolo_detection(yolo_model, image_path: str, conf_threshold: float = 0.25):
    """
    Run YOLO detection on an image.

    Returns:
        List of detections: [{box, class_id, confidence, class_name}, ...]
    """
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)[0]

    _NORM = {
        'fiber': 'fiber', 'fibre': 'fiber',
        'film': 'film',
        'fragment': 'fragment', 'frag': 'fragment',
    }

    def _resolve_name(cid: int) -> tuple[int, str]:
        raw = results.names.get(cid, '').lower().strip()
        name = _NORM.get(raw, None)
        if name is None:
            if cid < len(EFFNET_CLASS_NAMES):
                name = EFFNET_CLASS_NAMES[cid]
            else:
                name = 'fragment'
        norm_id = EFFNET_CLASS_NAMES.index(name)
        return norm_id, name

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        norm_id, class_name = _resolve_name(class_id)

        detections.append({
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'class_id': norm_id,
            'confidence': conf,
            'class_name': class_name,
        })

    return detections


# ============================================================================
# Micro Per-Type Mask R-CNN Loader
# ============================================================================

def load_per_type_maskrcnn_micro(device: torch.device,
                                  maskrcnn_dir: str = None) -> dict:
    """
    Load 3 type-specific binary Mask R-CNN models for MICRO images.

    Returns:
        {type_name: model} for each type whose .pth file exists.
    """
    models = {}
    for mp_type in EFFNET_CLASS_NAMES:
        if maskrcnn_dir:
            path = Path(maskrcnn_dir) / f'maskrcnn_{mp_type}' / f'maskrcnn_{mp_type}_best.pth'
            if not path.exists():
                path = Path(maskrcnn_dir) / f'maskrcnn_{mp_type}' / 'maskrcnn_best.pth'
        else:
            path = None
            for candidate in MICRO_PER_TYPE_MODEL_CANDIDATES[mp_type]:
                candidate_path = Path(candidate)
                if candidate_path.exists():
                    path = candidate_path
                    break
            if path is None:
                path = Path(MICRO_PER_TYPE_MODEL_CANDIDATES[mp_type][0])

        if path.exists():
            models[mp_type] = load_maskrcnn(str(path), device, num_classes=NUM_CLASSES_BINARY)
        else:
            print(f"[Micro Mask R-CNN] Per-type model not found: {path}")

    return models


# ============================================================================
# Full Micro Pipeline
# ============================================================================

def run_pipeline_micro(
    image_path: str,
    yolo_model_path: str = MICRO_YOLO_PATH,
    maskrcnn_model_path: str = MICRO_MASKRCNN_FALLBACK_PATH,
    maskrcnn_dir: str = None,
    effnet_model_path: str = MICRO_EFFNET_PATH,
    output_dir: str = 'experiments/pipeline_output_micro',
    yolo_conf: float = 0.25,
    mask_threshold: float = 0.5,
    use_maskrcnn: bool = True,
    use_effnet: bool = True,
    pixel_to_micron: float = 1.0,
    crop_padding: int = 30,
    nms_iou: float = 0.3,
):
    """
    Run the complete YOLO + EfficientNet + Mask R-CNN pipeline on MICRO images.

    Uses micro-specific models from experiments/micro/.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print("  MICROPLASTIC DETECTION PIPELINE — MICRO MODE")
    print(f"  YOLO -> NMS -> EfficientNet -> Type-Specific Mask R-CNN -> Size")
    print(f"{'='*70}")
    print(f"  Device         : {device}")
    print(f"  Image          : {image_path}")
    print(f"  YOLO model     : {yolo_model_path}")
    print(f"  EfficientNet   : {effnet_model_path if use_effnet else 'DISABLED'}")
    if use_maskrcnn:
        print(f"  Mask R-CNN     : per-type routing (fiber/film/fragment MICRO models)")
        if maskrcnn_model_path:
            print(f"  Mask R-CNN (fb): {maskrcnn_model_path} (fallback if per-type missing)")
    else:
        print(f"  Mask R-CNN     : DISABLED (ellipse fallback)")
    print(f"  YOLO conf      : {yolo_conf}")
    print(f"  NMS IoU        : {nms_iou}")
    print(f"  Pixel->um ratio: {pixel_to_micron}")
    print(f"{'='*70}\n")

    # ---- Load Models ----
    print("[1/6] Loading MICRO models...")
    yolo_model = YOLO(yolo_model_path)

    per_type_models: dict = {}
    fallback_maskrcnn = None
    if use_maskrcnn:
        per_type_models = load_per_type_maskrcnn_micro(device, maskrcnn_dir)
        if per_type_models:
            print(f"  Loaded {len(per_type_models)} per-type MICRO Mask R-CNN model(s): "
                  f"{list(per_type_models.keys())}")
        missing = [t for t in EFFNET_CLASS_NAMES if t not in per_type_models]
        if missing and maskrcnn_model_path and Path(maskrcnn_model_path).exists():
            fallback_maskrcnn = load_maskrcnn(maskrcnn_model_path, device,
                                             num_classes=NUM_CLASSES_MASKRCNN)
            print(f"  Fallback Mask R-CNN for: {missing}")
        elif missing and not per_type_models:
            print(f"  WARNING: No Mask R-CNN models found. Will use ellipse fallback.")

    effnet_model = None
    if use_effnet:
        effnet_model = load_effnet(effnet_model_path, device)
        if effnet_model is None:
            print("  -> EfficientNet not available, falling back to YOLO classes")
            use_effnet = False

    effnet_transform = get_effnet_transform() if use_effnet else None
    maskrcnn_transform = get_maskrcnn_transform() if use_maskrcnn else None

    # ---- Load Image ----
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = image.shape[:2]
    print(f"  Image size: {w} x {h}")

    # ---- Detect filter paper circle ----
    filter_circle = None
    full_coverage = False
    try:
        fc_center, fc_radius, _fc_mask, fc_method, full_coverage = \
            detect_filter_circle_from_array(image)
        if not full_coverage:
            filter_circle = (fc_center, fc_radius)
    except Exception as e:
        print(f"  Warning: filter circle detection failed: {e}")

    # ==================================================================
    # Stage 1: YOLO Detection
    # ==================================================================
    print(f"\n[2/6] YOLO detection (conf={yolo_conf})...")
    raw_detections = run_yolo_detection(yolo_model, image_path, yolo_conf)
    print(f"  Raw YOLO detections: {len(raw_detections)}")

    # ==================================================================
    # Stage 2: Class-Agnostic NMS
    # ==================================================================
    print(f"\n[3/6] Class-agnostic NMS (iou={nms_iou})...")
    nms_input = [
        [d['box'][0], d['box'][1], d['box'][2], d['box'][3],
         d['class_id'], d['confidence']]
        for d in raw_detections
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
    print(f"  After NMS: {len(detections)} unique microplastics "
          f"(removed {len(raw_detections) - len(detections)} duplicates)")

    # ==================================================================
    # Stage 2b: Filter detections to filter-paper region only
    # ==================================================================
    if filter_circle is not None and not full_coverage:
        fc_cx, fc_cy = filter_circle[0]
        fc_r = filter_circle[1]
        before = len(detections)
        inside = []
        for det in detections:
            bx = det['box']
            box_cx = (bx[0] + bx[2]) / 2
            box_cy = (bx[1] + bx[3]) / 2
            dist = np.sqrt((box_cx - fc_cx) ** 2 + (box_cy - fc_cy) ** 2)
            if dist <= fc_r:
                inside.append(det)
        detections = inside
        print(f"  Filter-paper ROI: kept {len(detections)}/{before} detections "
              f"(removed {before - len(detections)} outside filter)")

    # ==================================================================
    # Stage 3: EfficientNet Classification
    # ==================================================================
    print(f"\n[4/6] EfficientNet classification on {len(detections)} crops...")

    classified = []
    for i, det in enumerate(detections):
        box = det['box']
        crop, crop_box, rel_box = crop_detection(image, box, padding=crop_padding)

        if use_effnet and effnet_model is not None:
            effnet_class_id, effnet_class_name, effnet_conf, effnet_probs = \
                classify_crop(effnet_model, crop, device, effnet_transform)
            final_class_name = effnet_class_name
            final_class_id = effnet_class_id
            classification_source = 'effnet'
        else:
            final_class_name = det['class_name']
            final_class_id = det['class_id']
            effnet_conf = 0.0
            effnet_probs = None
            classification_source = 'yolo'

        classified.append({
            **det,
            'crop': crop,
            'crop_box': crop_box,
            'final_class_name': final_class_name,
            'final_class_id': final_class_id,
            'effnet_conf': effnet_conf,
            'effnet_probs': effnet_probs,
            'classification_source': classification_source,
        })

    reclassified = sum(1 for c in classified
                       if c['class_name'] != c['final_class_name'])
    if use_effnet:
        print(f"  EfficientNet reclassified {reclassified}/{len(classified)} detections")

    # ==================================================================
    # Stage 4: Type-Specific Mask R-CNN Segmentation
    # ==================================================================
    print(f"\n[5/6] Mask R-CNN segmentation (type-routed, MICRO)...")

    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    results = []

    for i, c in enumerate(classified):
        box = c['box']
        crop = c['crop']
        crop_box = c['crop_box']
        yolo_class_name = c['class_name']
        yolo_conf_score = c['confidence']
        final_class_name = c['final_class_name']
        final_class_id = c['final_class_id']
        effnet_conf = c['effnet_conf']
        effnet_probs = c['effnet_probs']
        classification_source = c['classification_source']

        if use_maskrcnn:
            chosen_model = per_type_models.get(final_class_name, fallback_maskrcnn)
            if chosen_model is not None:
                mask_crop, mask_conf = segment_crop(
                    chosen_model, crop, final_class_id, device,
                    maskrcnn_transform, mask_threshold
                )
                x1_pad, y1_pad, x2_pad, y2_pad = crop_box
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1_pad:y2_pad, x1_pad:x2_pad] = mask_crop
                model_name = final_class_name if final_class_name in per_type_models else 'generic'
                segmentation_source = f'maskrcnn_{model_name}'
            else:
                full_mask = generate_ellipse_mask(box, image.shape)
                mask_conf = 0.5
                segmentation_source = 'ellipse'
        else:
            full_mask = generate_ellipse_mask(box, image.shape)
            mask_conf = 0.5
            segmentation_source = 'ellipse'

        size_info = calculate_microplastic_size(full_mask, pixel_to_micron)

        color = COLORS.get(final_class_name, (255, 255, 255))
        mask_overlay[full_mask == 1] = color

        result = {
            'id': i + 1,
            'box': box,
            'yolo_class': yolo_class_name,
            'yolo_confidence': round(yolo_conf_score, 4),
            'final_class': final_class_name,
            'final_class_id': final_class_id,
            'classification_source': classification_source,
            'effnet_confidence': round(effnet_conf, 4),
            'effnet_probabilities': (
                {EFFNET_CLASS_NAMES[j]: round(float(p), 4) for j, p in enumerate(effnet_probs)}
                if effnet_probs is not None else None
            ),
            'mask_confidence': round(mask_conf, 4),
            'segmentation_source': segmentation_source,
            'size': size_info,
            'mask': full_mask,
        }
        results.append(result)

        changed = '*' if (use_effnet and yolo_class_name != final_class_name) else ''
        print(f"  [{i+1:3d}] {final_class_name:>8s} {changed:1s} | "
              f"YOLO={yolo_conf_score:.2f} ENet={effnet_conf:.2f} Mask={mask_conf:.2f} | "
              f"Area={size_info['area_px']}px  L={size_info['length_px']}px  "
              f"W={size_info['width_px']}px")

    # ---- Stage 5: Size Summary ----
    print(f"\n[6/6] Size analysis...")
    _print_size_summary(results, pixel_to_micron)
    vis_image = _create_visualization(image, results, mask_overlay, pixel_to_micron,
                                       filter_circle=filter_circle)

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    vis_path = Path(output_dir) / f"{stem}_pipeline_micro.png"
    cv2.imwrite(str(vis_path), vis_image)
    print(f"  Visualization: {vis_path}")

    mask_path = Path(output_dir) / f"{stem}_masks_micro.png"
    cv2.imwrite(str(mask_path), mask_overlay)

    report = {
        'image': str(image_path),
        'pipeline_mode': 'micro',
        'image_size': {'width': w, 'height': h},
        'filter_circle': {
            'center': list(filter_circle[0]),
            'radius': filter_circle[1],
        } if filter_circle else None,
        'pixel_to_micron': pixel_to_micron,
        'models': {
            'yolo': str(yolo_model_path),
            'effnet': str(effnet_model_path) if use_effnet else None,
            'maskrcnn_per_type': {
                t: str(next(
                    (Path(c) for c in MICRO_PER_TYPE_MODEL_CANDIDATES[t] if Path(c).exists()),
                    Path(MICRO_PER_TYPE_MODEL_CANDIDATES[t][0]),
                ))
                for t in per_type_models
            } if per_type_models else None,
            'maskrcnn_fallback': str(maskrcnn_model_path) if fallback_maskrcnn else None,
        },
        'total_detections': len(results),
        'counts': {
            name: sum(1 for r in results if r['final_class'] == name)
            for name in EFFNET_CLASS_NAMES
        },
        'detections': [
            {k: v for k, v in r.items() if k != 'mask'}
            for r in results
        ],
    }

    report_path = Path(output_dir) / f"{stem}_report_micro.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {report_path}")

    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE — MICRO MODE")
    print(f"{'='*70}")
    print(f"  Total detections: {len(results)}")
    for name in EFFNET_CLASS_NAMES:
        count = report['counts'][name]
        if count > 0:
            sizes = [r['size']['length_px'] for r in results if r['final_class'] == name]
            avg_len = np.mean(sizes) if sizes else 0
            print(f"    {name:>8s}: {count:3d}  "
                  f"(avg length: {avg_len:.1f}px / {avg_len * pixel_to_micron:.1f}um)")
    print(f"{'='*70}\n")

    return {
        'detections': results,
        'report': report,
        'visualization_path': str(vis_path),
        'report_path': str(report_path),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Microplastic Detection Pipeline (MICRO MODE): '
                    'YOLO + EfficientNet + Mask R-CNN + Size Calculation')

    parser.add_argument('--image', type=str, required=True,
                        help='Input image path or directory of images '
                             '(png/jpg/jpeg/tif/bmp)')

    parser.add_argument('--yolo', type=str,
                        default=MICRO_YOLO_PATH,
                        help='YOLO model path (micro)')
    parser.add_argument('--maskrcnn', type=str,
                        default=MICRO_MASKRCNN_FALLBACK_PATH,
                        help='Fallback single Mask R-CNN model path')
    parser.add_argument('--maskrcnn-dir', type=str, default=None,
                        help='Directory containing maskrcnn_fiber/, maskrcnn_film/, '
                             'maskrcnn_fragment/ sub-dirs for micro models')
    parser.add_argument('--effnet', type=str,
                        default=MICRO_EFFNET_PATH,
                        help='EfficientNet model path (micro)')

    parser.add_argument('--output', type=str,
                        default='experiments/pipeline_output_micro',
                        help='Output directory')

    parser.add_argument('--yolo-conf', type=float, default=0.25,
                        help='YOLO confidence threshold')
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                        help='Mask binarization threshold')
    parser.add_argument('--nms-iou', type=float, default=0.3,
                        help='IoU threshold for class-agnostic NMS')
    parser.add_argument('--padding', type=int, default=30,
                        help='Crop padding (pixels)')

    parser.add_argument('--no-maskrcnn', action='store_true',
                        help='Skip Mask R-CNN, use ellipse masks')
    parser.add_argument('--no-effnet', action='store_true',
                        help='Skip EfficientNet, use YOLO classes')

    parser.add_argument('--pixel-to-micron', type=float, default=1.0,
                        help='Microns per pixel for real-world size measurement.')

    args = parser.parse_args()

    input_path = Path(args.image)
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    if input_path.is_dir():
        image_files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        if not image_files:
            print(f"No images found in directory: {input_path}")
            return
        print(f"Directory mode: found {len(image_files)} images in '{input_path}'")
    elif input_path.is_file():
        image_files = [input_path]
    else:
        print(f"Error: '{args.image}' is not a file or directory.")
        return

    all_results = []
    for img_path in image_files:
        result = run_pipeline_micro(
            image_path=str(img_path),
            yolo_model_path=args.yolo,
            maskrcnn_model_path=args.maskrcnn,
            maskrcnn_dir=args.maskrcnn_dir,
            effnet_model_path=args.effnet,
            output_dir=args.output,
            yolo_conf=args.yolo_conf,
            mask_threshold=args.mask_threshold,
            use_maskrcnn=not args.no_maskrcnn,
            use_effnet=not args.no_effnet,
            pixel_to_micron=args.pixel_to_micron,
            crop_padding=args.padding,
            nms_iou=args.nms_iou,
        )
        all_results.append((str(img_path), result))

    if len(image_files) > 1:
        total = sum(r['report']['total_detections'] for _, r in all_results)
        print(f"\n{'='*70}")
        print(f"  BATCH SUMMARY (MICRO): {len(image_files)} images, {total} total detections")
        print(f"{'='*70}")
        for img_path, r in all_results:
            n = r['report']['total_detections']
            counts = r['report']['counts']
            cnt_str = '  '.join(f"{k}:{v}" for k, v in counts.items() if v > 0)
            print(f"  {Path(img_path).name:30s}: {n:3d} detections  {cnt_str}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
