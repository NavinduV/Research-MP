"""
YOLO + EfficientNet + Per-Type Mask R-CNN Pipeline for Microplastic Detection,
Classification & Segmentation.

Complete pipeline architecture:
    1. YOLO: Fast detection -> bounding boxes
    2. Class-agnostic NMS -> deduplicate overlapping boxes
    3. EfficientNet: Refined classification (fiber/film/fragment)
    4. Route each crop to its TYPE-SPECIFIC Mask R-CNN for segmentation:
         fiber    → maskrcnn_fiber   (binary: bg + fiber)
         film     → maskrcnn_film    (binary: bg + film)
         fragment → maskrcnn_fragment (binary: bg + fragment)
    5. Size calculation: Compute actual microplastic dimensions from masks

Usage:
    # Full pipeline (auto-discovers per-type models)
    python src/pipeline/pipeline_inference.py --image path/to/image.png

    # With custom per-type model directory
    python src/pipeline/pipeline_inference.py --image path/to/image.png \\
        --maskrcnn-dir experiments

    # Fallback: single generic Mask R-CNN (old behaviour)
    python src/pipeline/pipeline_inference.py --image path/to/image.png \\
        --maskrcnn experiments/maskrcnn/maskrcnn_crops_best.pth
    # Skip Mask R-CNN (use ellipse masks, faster)
    python src/pipeline/pipeline_inference.py --image path/to/image.png --no-maskrcnn

    # Skip EfficientNet (use YOLO classes)
    python src/pipeline/pipeline_inference.py --image path/to/image.png --no-effnet

    # Set pixel-to-micron ratio for real-world size
    python src/pipeline/pipeline_inference.py --image path/to/image.png --pixel-to-micron 2.5
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


# ============================================================================
# Configuration
# ============================================================================

NUM_CLASSES_BINARY = 2    # per-type models: background + 1 type
NUM_CLASSES_MASKRCNN = 4  # legacy single model: background + fiber + film + fragment
NUM_CLASSES_EFFNET = 3    # fiber, film, fragment

CLASS_NAMES = ['background', 'fiber', 'film', 'fragment']
EFFNET_CLASS_NAMES = ['fiber', 'film', 'fragment']

YOLO_TO_MASKRCNN_CLASS = {0: 1, 1: 2, 2: 3}  # fiber=0->1, film=1->2, fragment=2->3
MASKRCNN_TO_CLASS_NAME = {1: 'fiber', 2: 'film', 3: 'fragment'}

# Default per-type model path candidates (new macro layout first, legacy layout second)
# NOTE: train_maskrcnn_per_type.py saves weights as "maskrcnn_best.pth" (generic name),
# so the correct filename is maskrcnn_best.pth — NOT maskrcnn_{type}_best.pth.
PER_TYPE_MODEL_CANDIDATES = {
    'fiber': [
        'experiments/macro/maskrcnn_fiber/maskrcnn_best.pth',
        'experiments/macro/maskrcnn_fiber/maskrcnn_fiber_best.pth',
        'experiments/maskrcnn_fiber/maskrcnn_best.pth',
        'experiments/maskrcnn_fiber/maskrcnn_fiber_best.pth',
    ],
    'film': [
        'experiments/macro/maskrcnn_film/maskrcnn_best.pth',
        'experiments/macro/maskrcnn_film/maskrcnn_film_best.pth',
        'experiments/maskrcnn_film/maskrcnn_best.pth',
        'experiments/maskrcnn_film/maskrcnn_film_best.pth',
    ],
    'fragment': [
        'experiments/macro/maskrcnn_fragment/maskrcnn_best.pth',
        'experiments/macro/maskrcnn_fragment/maskrcnn_fragment_best.pth',
        'experiments/maskrcnn_fragment/maskrcnn_best.pth',
        'experiments/maskrcnn_fragment/maskrcnn_fragment_best.pth',
    ],
}

# Colors for visualization (BGR format for OpenCV)
COLORS = {
    'fiber':          (0, 0, 255),     # red
    'film':           (0, 255, 255),   # yellow
    'fragment':       (0, 255, 0),     # green
    'microplastic':   (255, 0, 0),     # blue — generic label when EfficientNet is off
}


# ============================================================================
# Class-Agnostic NMS (ported from train_yolo.py)
# ============================================================================

def class_agnostic_nms(boxes, iou_threshold=0.3):
    """
    Class-agnostic Non-Maximum Suppression.

    Keeps only the highest-confidence detection per physical location,
    regardless of predicted class.  This ensures each real microplastic
    produces exactly ONE bounding box.

    Args:
        boxes: list/array  [[x1, y1, x2, y2, cls_id, conf], ...]
        iou_threshold: IoU above which the lower-confidence box is suppressed.
    Returns:
        kept: list of boxes that survived NMS (same format as input rows)
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 5]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        min_area = np.minimum(areas[i], areas[order[1:]])
        containment = inter / np.maximum(min_area, 1e-6)

        inds = np.where((iou <= iou_threshold) & (containment <= 0.6))[0]
        order = order[inds + 1]

    return boxes[keep].tolist()


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


def load_maskrcnn(model_path: str, device: torch.device, num_classes: int = NUM_CLASSES_MASKRCNN):
    """Load trained Mask R-CNN model (works for both single-model and per-type)."""
    model = get_maskrcnn_model(num_classes)
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Mask R-CNN] Loaded from: {model_path}")
    else:
        print(f"[Mask R-CNN] WARNING: Not found at {model_path}, using random weights")
    model.to(device)
    model.eval()
    return model


def load_per_type_maskrcnn(device: torch.device,
                           maskrcnn_dir: str = None) -> dict:
    """
    Load 3 type-specific binary Mask R-CNN models.

    Returns:
        {type_name: model} for each type whose .pth file exists.
        Returns empty dict if no per-type models found.
    """
    models = {}
    for mp_type in EFFNET_CLASS_NAMES:
        if maskrcnn_dir:
            # Custom dir: look for maskrcnn_{type}/maskrcnn_{type}_best.pth
            path = Path(maskrcnn_dir) / f'maskrcnn_{mp_type}' / f'maskrcnn_{mp_type}_best.pth'
            if not path.exists():
                # Also try maskrcnn_{type}/maskrcnn_best.pth
                path = Path(maskrcnn_dir) / f'maskrcnn_{mp_type}' / 'maskrcnn_best.pth'
        else:
            path = None
            for candidate in PER_TYPE_MODEL_CANDIDATES[mp_type]:
                candidate_path = Path(candidate)
                if candidate_path.exists():
                    path = candidate_path
                    break
            if path is None:
                # Keep a deterministic path in logs when none of the candidates exist.
                path = Path(PER_TYPE_MODEL_CANDIDATES[mp_type][0])

        if path.exists():
            models[mp_type] = load_maskrcnn(str(path), device, num_classes=NUM_CLASSES_BINARY)
        else:
            print(f"[Mask R-CNN] Per-type model not found: {path}")

    return models


def load_effnet(model_path: str, device: torch.device):
    """Load trained EfficientNet classifier."""
    try:
        model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=NUM_CLASSES_EFFNET)
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"[EfficientNet] WARNING: Not found at {model_path}")
            return None
    except Exception as e:
        # Fallback to B0 if B3 fails
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES_EFFNET)
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"[EfficientNet] WARNING: Not found at {model_path}")
            return None

    if Path(model_path).exists():
        print(f"[EfficientNet] Loaded from: {model_path}")
    
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
    """Mask R-CNN preprocessing transform for crops.

    IMPORTANT: Must match the training transform in train_maskrcnn_per_type.py
    which uses A.Resize(CROP_SIZE, CROP_SIZE) with CROP_SIZE=128.
    Using a different resize strategy (e.g. LongestMaxSize+Pad) causes the
    model to produce degraded, oval-shaped masks instead of precise contours.
    """
    return A.Compose([
        A.Resize(128, 128),
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

    # Build a safe name lookup from the model's own class dict
    # e.g. {0: 'fiber', 1: 'film', 2: 'fragment'} or whatever the model reports.
    # Then normalise to the three known EfficientNet class names so downstream
    # code stays consistent regardless of how the model was exported.
    _NORM = {
        'fiber': 'fiber', 'fibre': 'fiber',
        'film': 'film',
        'fragment': 'fragment', 'frag': 'fragment',
    }

    def _resolve_name(cid: int) -> tuple[int, str]:
        """Return (normalised_class_id, class_name) for a YOLO class index."""
        raw = results.names.get(cid, '').lower().strip()
        name = _NORM.get(raw, None)
        if name is None:
            # Fallback: map by position if within range, else default to 'fragment'
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

def _refine_mask(mask_binary: np.ndarray, crop_shape: tuple) -> np.ndarray:
    """Apply morphological refinement to produce clean, contour-following masks.

    Steps:
        1. Close small gaps in the mask (morphological closing)
        2. Remove small noise blobs
        3. Keep only the largest connected component
        4. Smooth jagged edges with a small Gaussian blur + re-threshold
    """
    h, w = crop_shape[:2]

    # Scale kernel size relative to the mask dimensions (at least 3)
    ksize = max(3, min(h, w) // 30)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # Close small holes / gaps
    refined = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Open to remove small noise specks
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    if num_labels > 1:
        # Component 0 is background, find largest foreground component
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
        largest_label = 1 + int(np.argmax(areas))
        refined = (labels == largest_label).astype(np.uint8)

    # Smooth jagged edges: blur + re-threshold
    smooth_ksize = max(3, min(h, w) // 20)
    if smooth_ksize % 2 == 0:
        smooth_ksize += 1
    smoothed = cv2.GaussianBlur(refined.astype(np.float32),
                                 (smooth_ksize, smooth_ksize), 0)
    refined = (smoothed > 0.4).astype(np.uint8)

    return refined


def segment_crop(maskrcnn_model, crop_bgr: np.ndarray, class_id: int, device: torch.device,
                 transform, mask_threshold: float = 0.5):
    """
    Run Mask R-CNN **purely for foreground segmentation** on a cropped detection.

    The crop already contains a confirmed microplastic (validated by YOLO + EfficientNet).
    Mask R-CNN is used ONLY to produce a pixel-level mask — it does NOT filter or
    re-validate the detection.

    Strategy:
        1. Run Mask R-CNN on the crop
        2. Merge high-confidence masks into one combined foreground mask
           (class-agnostic — we already know the class from EfficientNet)
        3. Resize the SOFT probability map first, THEN binarize (preserves
           contour detail that would be lost by binarize-then-resize)
        4. Apply morphological refinement for clean contour-following masks
        5. If Mask R-CNN finds nothing, fall back to an ellipse mask

    Returns:
        mask: Binary mask for the crop (H, W)
        confidence: Best mask confidence score (for metadata only)
    """
    crop_h, crop_w = crop_bgr.shape[:2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    transformed = transform(image=crop_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = maskrcnn_model(input_tensor)[0]

    masks = predictions['masks'].cpu().numpy()   # (N, 1, H_model, W_model) float
    scores = predictions['scores'].cpu().numpy()  # (N,)

    # ---- Class-agnostic mask merging ----
    # We trust YOLO+EfficientNet for classification.
    # From Mask R-CNN we only want foreground mask(s).
    #
    # Merge all masks with score >= 0.3 (weighted by score) so that
    # multiple overlapping predictions combine into better coverage.
    # Then use the best score as the reported confidence.
    MIN_MERGE_SCORE = 0.3

    combined_soft = None
    best_score = 0.0

    if len(masks) > 0:
        best_score = float(scores.max())
        # Filter to masks above the merge threshold
        good_idxs = np.where(scores >= MIN_MERGE_SCORE)[0]
        if len(good_idxs) > 0:
            # Weighted merge: combine soft masks weighted by their scores
            combined_soft = np.zeros_like(masks[0][0], dtype=np.float32)
            total_weight = 0.0
            for idx in good_idxs:
                w_score = float(scores[idx])
                combined_soft += masks[idx][0] * w_score
                total_weight += w_score
            combined_soft /= max(total_weight, 1e-6)

    # Fallback: ellipse mask when Mask R-CNN produces nothing useful
    if combined_soft is None or combined_soft.max() < 0.1:
        combined_soft = np.zeros((crop_h, crop_w), dtype=np.float32)
        center = (crop_w // 2, crop_h // 2)
        axes = (max(1, crop_w // 2 - 2), max(1, crop_h // 2 - 2))
        cv2.ellipse(combined_soft, center, axes, 0, 0, 360, 1.0, -1)
        best_score = 0.5
        # Binarize directly (already at crop size)
        mask_final = (combined_soft > mask_threshold).astype(np.uint8)
        return mask_final, best_score

    # ---- Resize SOFT mask to crop dimensions BEFORE binarizing ----
    # This preserves contour detail that would be destroyed by
    # binarize → INTER_NEAREST resize (which produced blocky/oval masks).
    soft_resized = cv2.resize(combined_soft, (crop_w, crop_h),
                               interpolation=cv2.INTER_LINEAR)

    # Binarize the high-resolution soft mask
    mask_binary = (soft_resized > mask_threshold).astype(np.uint8)

    # ---- Morphological refinement ----
    mask_refined = _refine_mask(mask_binary, crop_bgr.shape)

    # Sanity: if refinement wiped the mask, fall back to the raw binary
    if mask_refined.sum() < 10:
        mask_refined = mask_binary

    return mask_refined, best_score


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
    maskrcnn_dir: str = None,
    effnet_model_path: str = 'experiments/efficientnet_best.pth',
    output_dir: str = 'experiments/pipeline_output',
    yolo_conf: float = 0.25,
    mask_threshold: float = 0.5,
    use_maskrcnn: bool = True,
    use_effnet: bool = True,
    pixel_to_micron: float = 1.0,
    crop_padding: int = 30,
    nms_iou: float = 0.3,
):
    """
    Run the complete YOLO + EfficientNet + Mask R-CNN pipeline.

    Type-routing (default):
        After EfficientNet classifies each crop, the crop is sent to the
        corresponding type-specific binary Mask R-CNN model:
            fiber    → maskrcnn_fiber_best.pth
            film     → maskrcnn_film_best.pth
            fragment → maskrcnn_fragment_best.pth

        If per-type models are not found, falls back to the single
        generic Mask R-CNN (maskrcnn_model_path).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print("  MICROPLASTIC DETECTION PIPELINE")
    print(f"  YOLO -> NMS -> EfficientNet -> Type-Specific Mask R-CNN -> Size")
    print(f"{'='*70}")
    print(f"  Device         : {device}")
    print(f"  Image          : {image_path}")
    print(f"  YOLO model     : {yolo_model_path}")
    print(f"  EfficientNet   : {effnet_model_path if use_effnet else 'DISABLED'}")
    if use_maskrcnn:
        print(f"  Mask R-CNN     : per-type routing (fiber/film/fragment models)")
        if maskrcnn_model_path:
            print(f"  Mask R-CNN (fb): {maskrcnn_model_path} (fallback if per-type missing)")
    else:
        print(f"  Mask R-CNN     : DISABLED (ellipse fallback)")
    print(f"  YOLO conf      : {yolo_conf}")
    print(f"  NMS IoU        : {nms_iou}")
    print(f"  Pixel->um ratio: {pixel_to_micron}")
    print(f"{'='*70}\n")

    # ---- Load Models ----
    print("[1/6] Loading models...")
    yolo_model = YOLO(yolo_model_path)

    # Per-type Mask R-CNN models (preferred)
    per_type_models: dict = {}
    fallback_maskrcnn = None
    if use_maskrcnn:
        per_type_models = load_per_type_maskrcnn(device, maskrcnn_dir)
        if per_type_models:
            print(f"  Loaded {len(per_type_models)} per-type Mask R-CNN model(s): "
                  f"{list(per_type_models.keys())}")
        # Load single model as fallback for types without a per-type model
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

    # Transforms
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
    # Stage 2: Class-Agnostic NMS  (same as train_yolo.py predict)
    # ==================================================================
    print(f"\n[3/6] Class-agnostic NMS (iou={nms_iou})...")
    nms_input = [
        [d['box'][0], d['box'][1], d['box'][2], d['box'][3],
         d['class_id'], d['confidence']]
        for d in raw_detections
    ]
    nms_kept = class_agnostic_nms(nms_input, iou_threshold=nms_iou)

    # Rebuild detection dicts from NMS survivors
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
    # Stage 3: EfficientNet Classification  (class refinement only)
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
            # Without EfficientNet, label as generic "microplastic"
            final_class_name = 'microplastic'
            final_class_id = -1
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
    print(f"\n[5/6] Mask R-CNN segmentation (type-routed)...")

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

        # ---- Route to type-specific model (or fallback) ----
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

    # ---- Stage 5: Size Summary ----
    print(f"\n[6/6] Size analysis...")
    _print_size_summary(results, pixel_to_micron)
    vis_image = _create_visualization(image, results, mask_overlay, pixel_to_micron,
                                       filter_circle=filter_circle,
                                       use_maskrcnn=use_maskrcnn)

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
                    (Path(c) for c in PER_TYPE_MODEL_CANDIDATES[t] if Path(c).exists()),
                    Path(PER_TYPE_MODEL_CANDIDATES[t][0]),
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


def _create_visualization(image, results, mask_overlay, pixel_to_micron,
                          filter_circle=None, use_maskrcnn=True):
    """Create annotated visualization image.

    Args:
        image: Original BGR image.
        results: List of detection result dicts.
        mask_overlay: Colour mask overlay (H, W, 3).
        pixel_to_micron: Scale factor.
        filter_circle: Optional (center, radius) tuple for the filter paper.
        use_maskrcnn: If False, skip mask overlay blend — show only boxes + labels.
    """
    vis = image.copy()

    # Draw the filter-paper circle first (underneath everything else)
    if filter_circle is not None:
        center, radius = filter_circle
        cv2.circle(vis, center, radius, (0, 200, 0), 2, cv2.LINE_AA)

    # Blend mask overlay (semi-transparent) — only when Mask R-CNN is enabled
    if use_maskrcnn:
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
                        default='experiments/maskrcnn/maskrcnn_crops_best.pth',
                        help='Fallback single Mask R-CNN model path '
                             '(used when per-type model is missing)')
    parser.add_argument('--maskrcnn-dir', type=str, default=None,
                        help='Directory containing maskrcnn_fiber/, maskrcnn_film/, '
                             'maskrcnn_fragment/ sub-dirs (default: experiments/)')
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
    parser.add_argument('--nms-iou', type=float, default=0.3,
                        help='IoU threshold for class-agnostic NMS (0.0-1.0)')
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
