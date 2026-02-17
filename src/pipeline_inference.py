"""
YOLO + EfficientNet + Mask R-CNN Pipeline for Microplastic Detection,
Classification & Segmentation.

Complete pipeline architecture:
    1. YOLO: Fast detection -> bounding boxes
    2. Crop each detection with padding
    3. For each crop (in parallel):
       - EfficientNet: Refined classification (fiber/film/fragment)
       - Mask R-CNN: Precise pixel-level segmentation mask
    4. Size calculation: Compute actual microplastic dimensions from masks

Usage:
    # Full pipeline (YOLO + EfficientNet + Mask R-CNN + Size)
    python src/pipeline_inference.py --image path/to/image.png

    # With custom models
    python src/pipeline_inference.py --image path/to/image.png \\
        --yolo experiments/microplastic_yolo/weights/best.pt \\
        --maskrcnn experiments/maskrcnn_crops_best.pth \\
        --effnet experiments/efficientnet_best.pth

    # Skip Mask R-CNN (use ellipse masks, faster)
    python src/pipeline_inference.py --image path/to/image.png --no-maskrcnn

    # Skip EfficientNet (use YOLO classes)
    python src/pipeline_inference.py --image path/to/image.png --no-effnet

    # Set pixel-to-micron ratio for real-world size
    python src/pipeline_inference.py --image path/to/image.png --pixel-to-micron 2.5
"""

import argparse
import json
import cv2
import numpy as np
import torch
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


# ============================================================================
# Configuration
# ============================================================================

NUM_CLASSES_MASKRCNN = 4  # background + fiber + film + fragment
NUM_CLASSES_EFFNET = 3    # fiber, film, fragment

CLASS_NAMES = ['background', 'fiber', 'film', 'fragment']
EFFNET_CLASS_NAMES = ['fiber', 'film', 'fragment']

YOLO_TO_MASKRCNN_CLASS = {0: 1, 1: 2, 2: 3}  # fiber=0->1, film=1->2, fragment=2->3
MASKRCNN_TO_CLASS_NAME = {1: 'fiber', 2: 'film', 3: 'fragment'}

# Colors for visualization (BGR format for OpenCV)
COLORS = {
    'fiber':    (0, 0, 255),     # red
    'film':     (0, 255, 255),   # yellow
    'fragment': (0, 255, 0),     # green
}


# ============================================================================
# Model Loaders
# ============================================================================

def get_maskrcnn_model(num_classes: int):
    """Create Mask R-CNN model with custom number of classes."""
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def load_maskrcnn(model_path: str, device: torch.device):
    """Load trained Mask R-CNN model."""
    model = get_maskrcnn_model(NUM_CLASSES_MASKRCNN)
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Mask R-CNN] Loaded from: {model_path}")
    else:
        print(f"[Mask R-CNN] WARNING: Not found at {model_path}, using random weights")
    model.to(device)
    model.eval()
    return model


def load_effnet(model_path: str, device: torch.device):
    """Load trained EfficientNet-B0 classifier."""
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES_EFFNET)
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[EfficientNet] Loaded from: {model_path}")
    else:
        print(f"[EfficientNet] WARNING: Not found at {model_path}")
        return None
    model.to(device)
    model.eval()
    return model


# ============================================================================
# Preprocessing
# ============================================================================

def get_effnet_transform():
    """EfficientNet preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_maskrcnn_transform():
    """Mask R-CNN preprocessing transform for crops."""
    return A.Compose([
        A.LongestMaxSize(max_size=256),
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, fill=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ============================================================================
# Detection Stage (YOLO)
# ============================================================================

def run_yolo_detection(yolo_model, image_path: str, conf_threshold: float = 0.25):
    """
    Run YOLO detection on an image.

    Returns:
        List of detections: [{box, class_id, confidence, class_name}, ...]
    """
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())

        detections.append({
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'class_id': class_id,
            'confidence': conf,
            'class_name': EFFNET_CLASS_NAMES[class_id]
        })

    return detections


# ============================================================================
# Crop Extraction
# ============================================================================

def crop_detection(image: np.ndarray, box: list, padding: int = 20):
    """
    Crop a detection region from the image with padding.

    Returns:
        crop: Cropped image (BGR)
        crop_box: Padded box [x1, y1, x2, y2] in original image coords
        rel_box: Original bbox relative to crop
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)

    crop = image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
    crop_box = [x1_pad, y1_pad, x2_pad, y2_pad]
    rel_box = [x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad]

    return crop, crop_box, rel_box


# ============================================================================
# Classification Stage (EfficientNet)
# ============================================================================

def classify_crop(effnet_model, crop_bgr: np.ndarray, device: torch.device, transform):
    """
    Classify a crop using EfficientNet.

    Returns:
        class_id, class_name, confidence, probabilities
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = effnet_model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()

    return pred_class, EFFNET_CLASS_NAMES[pred_class], float(probs[pred_class]), probs.cpu().numpy()


# ============================================================================
# Segmentation Stage (Mask R-CNN)
# ============================================================================

def segment_crop(maskrcnn_model, crop_bgr: np.ndarray, class_id: int, device: torch.device,
                 transform, mask_threshold: float = 0.5):
    """
    Run Mask R-CNN segmentation on a cropped detection.

    Returns:
        mask: Binary mask for the crop (H, W)
        confidence: Mask confidence score
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    transformed = transform(image=crop_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = maskrcnn_model(input_tensor)[0]

    masks = predictions['masks'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    # Map to Mask R-CNN class
    target_class = YOLO_TO_MASKRCNN_CLASS.get(class_id, 1)

    # Find highest scoring mask of the target class
    best_mask = None
    best_score = 0.0

    for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        if label == target_class and score > best_score:
            best_mask = mask[0]
            best_score = score

    # If no matching class found, use highest scoring mask regardless
    if best_mask is None and len(masks) > 0:
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx][0]
        best_score = float(scores[best_idx])

    # Fallback: ellipse mask
    if best_mask is None:
        best_mask = np.zeros((256, 256), dtype=np.float32)
        cv2.ellipse(best_mask, (128, 128), (100, 60), 0, 0, 360, 1.0, -1)
        best_score = 0.5

    # Binarize and resize to crop size
    mask_binary = (best_mask > mask_threshold).astype(np.uint8)
    mask_resized = cv2.resize(mask_binary, (crop_bgr.shape[1], crop_bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    return mask_resized, best_score


def generate_ellipse_mask(box: list, image_shape: tuple):
    """Generate an ellipse mask from bounding box (fallback when no Mask R-CNN)."""
    x1, y1, x2, y2 = box
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = ((x2 - x1) // 2, (y2 - y1) // 2)

    if axes[0] > 0 and axes[1] > 0:
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)

    return mask


# ============================================================================
# Size Calculation from Mask
# ============================================================================

def calculate_microplastic_size(mask: np.ndarray, pixel_to_micron: float = 1.0):
    """
    Calculate physical dimensions of a microplastic from its binary mask.

    Uses contour analysis + rotated bounding box / fitted ellipse for
    accurate measurement of length, width, area, perimeter and circularity.

    Args:
        mask: Binary mask (H, W) with 1 = microplastic
        pixel_to_micron: Conversion factor (microns per pixel).
                         Set based on your imaging setup.

    Returns:
        dict with area, length, width, aspect_ratio, perimeter, circularity,
        centroid (all in both px and um units).
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    empty = {
        'area_px': 0, 'area_um2': 0.0,
        'length_px': 0, 'length_um': 0.0,
        'width_px': 0, 'width_um': 0.0,
        'aspect_ratio': 0.0,
        'perimeter_px': 0, 'perimeter_um': 0.0,
        'circularity': 0.0,
        'centroid': (0, 0),
    }
    if not contours:
        return empty

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Area
    area_px = cv2.contourArea(contour)
    area_um2 = area_px * (pixel_to_micron ** 2)

    # Perimeter
    perimeter_px = cv2.arcLength(contour, closed=True)
    perimeter_um = perimeter_px * pixel_to_micron

    # Rotated bounding box / fitted ellipse -> major & minor axis
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        _center, (minor_axis, major_axis), _angle = ellipse
        length_px = max(major_axis, minor_axis)
        width_px = min(major_axis, minor_axis)
    else:
        rect = cv2.minAreaRect(contour)
        _center, (w_rect, h_rect), _angle = rect
        length_px = max(w_rect, h_rect)
        width_px = min(w_rect, h_rect)

    length_um = length_px * pixel_to_micron
    width_um = width_px * pixel_to_micron

    # Aspect ratio
    aspect_ratio = length_px / max(width_px, 1e-6)

    # Circularity: 4*pi*area / perimeter^2  (1.0 = perfect circle)
    circularity = 0.0
    if perimeter_px > 0:
        circularity = (4 * np.pi * area_px) / (perimeter_px ** 2)

    # Centroid
    M = cv2.moments(contour)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    return {
        'area_px': int(area_px),
        'area_um2': round(area_um2, 2),
        'length_px': round(length_px, 1),
        'length_um': round(length_um, 2),
        'width_px': round(width_px, 1),
        'width_um': round(width_um, 2),
        'aspect_ratio': round(aspect_ratio, 2),
        'perimeter_px': round(perimeter_px, 1),
        'perimeter_um': round(perimeter_um, 2),
        'circularity': round(circularity, 3),
        'centroid': (cx, cy),
    }


# ============================================================================
# Full Pipeline
# ============================================================================

def run_pipeline(
    image_path: str,
    yolo_model_path: str = 'experiments/microplastic_yolo/weights/best.pt',
    maskrcnn_model_path: str = 'experiments/maskrcnn_crops_best.pth',
    effnet_model_path: str = 'experiments/efficientnet_best.pth',
    output_dir: str = 'experiments/pipeline_output',
    yolo_conf: float = 0.25,
    mask_threshold: float = 0.5,
    use_maskrcnn: bool = True,
    use_effnet: bool = True,
    pixel_to_micron: float = 1.0,
    crop_padding: int = 30,
):
    """
    Run the complete YOLO + EfficientNet + Mask R-CNN pipeline.

    Pipeline flow:
        Image -> YOLO (detect) -> Crop each detection ->
            |-- EfficientNet -> refined classification (fiber/film/fragment)
            +-- Mask R-CNN   -> precise pixel mask
        -> Size calculation from mask -> Report
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print("  MICROPLASTIC DETECTION PIPELINE")
    print(f"  YOLO -> Crop -> EfficientNet (classify) + Mask R-CNN (segment) -> Size")
    print(f"{'='*70}")
    print(f"  Device         : {device}")
    print(f"  Image          : {image_path}")
    print(f"  YOLO model     : {yolo_model_path}")
    print(f"  EfficientNet   : {effnet_model_path if use_effnet else 'DISABLED'}")
    print(f"  Mask R-CNN     : {maskrcnn_model_path if use_maskrcnn else 'DISABLED (ellipse fallback)'}")
    print(f"  YOLO conf      : {yolo_conf}")
    print(f"  Pixel->um ratio: {pixel_to_micron}")
    print(f"{'='*70}\n")

    # ---- Load Models ----
    print("[1/5] Loading models...")
    yolo_model = YOLO(yolo_model_path)

    maskrcnn_model = None
    if use_maskrcnn:
        maskrcnn_model = load_maskrcnn(maskrcnn_model_path, device)

    effnet_model = None
    if use_effnet:
        effnet_model = load_effnet(effnet_model_path, device)
        if effnet_model is None:
            print("  -> EfficientNet not available, falling back to YOLO classes")
            use_effnet = False

    # Transforms
    effnet_transform = get_effnet_transform() if use_effnet else None
    maskrcnn_transform = get_maskrcnn_transform() if use_maskrcnn else None

    # ---- Load Image ----
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = image.shape[:2]
    print(f"  Image size: {w} x {h}")

    # ---- Stage 1: YOLO Detection ----
    print(f"\n[2/5] YOLO detection (conf={yolo_conf})...")
    detections = run_yolo_detection(yolo_model, image_path, yolo_conf)
    print(f"  Found {len(detections)} microplastics")

    # ---- Stage 2 & 3: Crop -> EfficientNet + Mask R-CNN ----
    print(f"\n[3/5] Processing {len(detections)} crops (EfficientNet + Mask R-CNN)...")

    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    results = []

    for i, det in enumerate(detections):
        box = det['box']
        yolo_class_id = det['class_id']
        yolo_class_name = det['class_name']
        yolo_conf_score = det['confidence']

        # Crop the detection
        crop, crop_box, rel_box = crop_detection(image, box, padding=crop_padding)

        # ---- EfficientNet Classification ----
        if use_effnet and effnet_model is not None:
            effnet_class_id, effnet_class_name, effnet_conf, effnet_probs = \
                classify_crop(effnet_model, crop, device, effnet_transform)
            final_class_name = effnet_class_name
            final_class_id = effnet_class_id
            classification_source = 'effnet'
        else:
            final_class_name = yolo_class_name
            final_class_id = yolo_class_id
            effnet_conf = 0.0
            effnet_probs = None
            classification_source = 'yolo'

        # ---- Mask R-CNN Segmentation ----
        if use_maskrcnn and maskrcnn_model is not None:
            mask_crop, mask_conf = segment_crop(
                maskrcnn_model, crop, final_class_id, device,
                maskrcnn_transform, mask_threshold
            )
            # Place mask in original image coordinates
            x1_pad, y1_pad, x2_pad, y2_pad = crop_box
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y1_pad:y2_pad, x1_pad:x2_pad] = mask_crop
            segmentation_source = 'maskrcnn'
        else:
            full_mask = generate_ellipse_mask(box, image.shape)
            mask_conf = 0.5
            segmentation_source = 'ellipse'

        # ---- Size Calculation from Mask ----
        size_info = calculate_microplastic_size(full_mask, pixel_to_micron)

        # Apply color to mask overlay
        color = COLORS.get(final_class_name, (255, 255, 255))
        mask_overlay[full_mask == 1] = color

        result = {
            'id': i + 1,
            'box': box,
            # Classification
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
            # Segmentation
            'mask_confidence': round(mask_conf, 4),
            'segmentation_source': segmentation_source,
            # Size measurements
            'size': size_info,
            # Raw mask (not serialized to JSON)
            'mask': full_mask,
        }
        results.append(result)

        # Log
        changed = '*' if (use_effnet and yolo_class_name != final_class_name) else ''
        print(f"  [{i+1:3d}] {final_class_name:>8s} {changed:1s} | "
              f"YOLO={yolo_conf_score:.2f} ENet={effnet_conf:.2f} Mask={mask_conf:.2f} | "
              f"Area={size_info['area_px']}px  L={size_info['length_px']}px  "
              f"W={size_info['width_px']}px")

    # ---- Stage 4: Size Summary ----
    print(f"\n[4/5] Size analysis...")
    _print_size_summary(results, pixel_to_micron)

    # ---- Stage 5: Visualization ----
    print(f"\n[5/5] Creating visualization...")
    vis_image = _create_visualization(image, results, mask_overlay, pixel_to_micron)

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    vis_path = Path(output_dir) / f"{stem}_pipeline.png"
    cv2.imwrite(str(vis_path), vis_image)
    print(f"  Visualization: {vis_path}")

    mask_path = Path(output_dir) / f"{stem}_masks.png"
    cv2.imwrite(str(mask_path), mask_overlay)

    # Save JSON report (without binary mask data)
    report = {
        'image': str(image_path),
        'image_size': {'width': w, 'height': h},
        'pixel_to_micron': pixel_to_micron,
        'models': {
            'yolo': str(yolo_model_path),
            'effnet': str(effnet_model_path) if use_effnet else None,
            'maskrcnn': str(maskrcnn_model_path) if use_maskrcnn else None,
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

    report_path = Path(output_dir) / f"{stem}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report: {report_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE")
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
# Reporting Helpers
# ============================================================================

def _print_size_summary(results, pixel_to_micron):
    """Print a summary table of microplastic sizes by class."""
    if not results:
        print("  No detections to analyze.")
        return

    unit = 'um' if pixel_to_micron != 1.0 else 'px'
    scale = pixel_to_micron

    print(f"\n  {'Class':>10s} | {'Count':>5s} | {'Avg Length':>10s} | {'Avg Width':>10s} | "
          f"{'Avg Area':>12s} | {'Avg Circ.':>9s} | {'Avg AR':>6s}")
    print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*9}-+-{'-'*6}")

    for cls_name in EFFNET_CLASS_NAMES:
        class_results = [r for r in results if r['final_class'] == cls_name]
        if not class_results:
            continue

        n = len(class_results)
        avg_length = np.mean([r['size']['length_px'] * scale for r in class_results])
        avg_width = np.mean([r['size']['width_px'] * scale for r in class_results])
        avg_area = np.mean([r['size']['area_px'] * (scale ** 2) for r in class_results])
        avg_circ = np.mean([r['size']['circularity'] for r in class_results])
        avg_ar = np.mean([r['size']['aspect_ratio'] for r in class_results])

        print(f"  {cls_name:>10s} | {n:>5d} | {avg_length:>8.1f}{unit:>2s} | "
              f"{avg_width:>8.1f}{unit:>2s} | {avg_area:>10.1f}{unit}2 | "
              f"{avg_circ:>9.3f} | {avg_ar:>6.2f}")

    # Overall
    n = len(results)
    avg_length = np.mean([r['size']['length_px'] * scale for r in results])
    avg_area = np.mean([r['size']['area_px'] * (scale ** 2) for r in results])
    print(f"  {'TOTAL':>10s} | {n:>5d} | {avg_length:>8.1f}{unit:>2s} | "
          f"{'':>10s} | {avg_area:>10.1f}{unit}2 | {'':>9s} | {'':>6s}")


def _create_visualization(image, results, mask_overlay, pixel_to_micron):
    """Create annotated visualization image."""
    vis = image.copy()

    # Blend mask overlay (semi-transparent)
    vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

    unit = 'um' if pixel_to_micron != 1.0 else 'px'
    scale = pixel_to_micron

    for det in results:
        box = det['box']
        class_name = det['final_class']
        size = det['size']

        color = COLORS.get(class_name, (255, 255, 255))
        x1, y1, x2, y2 = box

        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label with class + size
        length_val = size['length_px'] * scale
        label = f"{class_name} {length_val:.0f}{unit}"

        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x1, y1 - label_h - 8),
                      (x1 + label_w + 5, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return vis


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Microplastic Detection Pipeline: '
                    'YOLO + EfficientNet + Mask R-CNN + Size Calculation')

    parser.add_argument('--image', type=str, required=True,
                        help='Input image path or directory of images '
                             '(png/jpg/jpeg/tif/bmp)')

    # Model paths
    parser.add_argument('--yolo', type=str,
                        default='experiments/microplastic_yolo/weights/best.pt',
                        help='YOLO model path')
    parser.add_argument('--maskrcnn', type=str,
                        default='experiments/maskrcnn_crops_best.pth',
                        help='Mask R-CNN model path')
    parser.add_argument('--effnet', type=str,
                        default='experiments/efficientnet_best.pth',
                        help='EfficientNet model path')

    # Output
    parser.add_argument('--output', type=str,
                        default='experiments/pipeline_output',
                        help='Output directory')

    # Thresholds
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                        help='YOLO confidence threshold')
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                        help='Mask binarization threshold')
    parser.add_argument('--padding', type=int, default=30,
                        help='Crop padding (pixels)')

    # Feature toggles
    parser.add_argument('--no-maskrcnn', action='store_true',
                        help='Skip Mask R-CNN, use ellipse masks (faster)')
    parser.add_argument('--no-effnet', action='store_true',
                        help='Skip EfficientNet, use YOLO classes')

    # Size calibration
    parser.add_argument('--pixel-to-micron', type=float, default=1.0,
                        help='Microns per pixel for real-world size measurement. '
                             'Calibrate with a stage micrometer.')

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
        result = run_pipeline(
            image_path=str(img_path),
            yolo_model_path=args.yolo,
            maskrcnn_model_path=args.maskrcnn,
            effnet_model_path=args.effnet,
            output_dir=args.output,
            yolo_conf=args.yolo_conf,
            mask_threshold=args.mask_threshold,
            use_maskrcnn=not args.no_maskrcnn,
            use_effnet=not args.no_effnet,
            pixel_to_micron=args.pixel_to_micron,
            crop_padding=args.padding,
        )
        all_results.append((str(img_path), result))

    if len(image_files) > 1:
        total = sum(r['report']['total_detections'] for _, r in all_results)
        print(f"\n{'='*70}")
        print(f"  BATCH SUMMARY: {len(image_files)} images, {total} total detections")
        print(f"{'='*70}")
        for img_path, r in all_results:
            n = r['report']['total_detections']
            counts = r['report']['counts']
            cnt_str = '  '.join(f"{k}:{v}" for k, v in counts.items() if v > 0)
            print(f"  {Path(img_path).name:30s}: {n:3d} detections  {cnt_str}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
