"""
YOLO + Mask R-CNN Two-Stage Pipeline for Microplastic Detection and Segmentation.

This pipeline combines the strengths of both models:
- YOLO: Fast and accurate object detection
- Mask R-CNN: Precise instance segmentation on detected regions

Flow:
1. YOLO detects microplastics and returns bounding boxes + classes
2. Each detected region is cropped with padding
3. Mask R-CNN segments each crop to produce precise masks
4. Masks are combined back into the original image space

Usage:
    python src/pipeline_inference.py --image path/to/image.png
    python src/pipeline_inference.py --image path/to/image.png --yolo experiments/microplastic_yolo/weights/best.pt
"""

import argparse
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


# ============================================================================
# Configuration
# ============================================================================

NUM_CLASSES = 4  # background + fiber + film + fragment
CLASS_NAMES = ['background', 'fiber', 'film', 'fragment']
YOLO_TO_MASKRCNN_CLASS = {0: 1, 1: 2, 2: 3}  # fiber=0->1, film=1->2, fragment=2->3

# Colors for visualization (BGR format for OpenCV)
COLORS = {
    1: (0, 0, 255),    # fiber - red
    2: (0, 255, 255),  # film - yellow
    3: (0, 255, 0),    # fragment - green
}


# ============================================================================
# Mask R-CNN Model
# ============================================================================

def get_maskrcnn_model(num_classes: int):
    """Create Mask R-CNN model with custom number of classes."""
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model


def load_maskrcnn_model(model_path: str, device: torch.device):
    """Load trained Mask R-CNN model."""
    model = get_maskrcnn_model(NUM_CLASSES)
    
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded Mask R-CNN from: {model_path}")
    else:
        print(f"Warning: Mask R-CNN model not found at {model_path}, using random weights")
    
    model.to(device)
    model.eval()
    return model


# ============================================================================
# Detection Stage (YOLO)
# ============================================================================

def run_yolo_detection(yolo_model, image_path: str, conf_threshold: float = 0.25):
    """
    Run YOLO detection on an image.
    
    Returns:
        List of detections, each with: box (x1,y1,x2,y2), class_id, confidence
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
            'class_name': CLASS_NAMES[YOLO_TO_MASKRCNN_CLASS[class_id]]
        })
    
    return detections


# ============================================================================
# Segmentation Stage (Mask R-CNN)
# ============================================================================

def crop_detection(image: np.ndarray, box: list, padding: int = 20):
    """
    Crop a detection region from the image with padding.
    
    Args:
        image: Original image (H, W, C)
        box: Bounding box [x1, y1, x2, y2]
        padding: Padding around the box
    
    Returns:
        crop: Cropped image
        crop_box: Adjusted box coordinates [x1, y1, x2, y2] in original image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    
    # Add padding
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)
    
    crop = image[y1_pad:y2_pad, x1_pad:x2_pad].copy()
    crop_box = [x1_pad, y1_pad, x2_pad, y2_pad]
    
    # Relative box within crop
    rel_box = [x1 - x1_pad, y1 - y1_pad, x2 - x1_pad, y2 - y1_pad]
    
    return crop, crop_box, rel_box


def segment_crop(maskrcnn_model, crop: np.ndarray, class_id: int, device: torch.device,
                 mask_threshold: float = 0.5):
    """
    Run Mask R-CNN segmentation on a cropped detection.
    
    Args:
        maskrcnn_model: Loaded Mask R-CNN model
        crop: Cropped image (RGB)
        class_id: Expected class ID (from YOLO)
        device: Torch device
        mask_threshold: Threshold for binarizing mask
    
    Returns:
        mask: Binary mask for the crop (H, W)
        confidence: Mask confidence score
    """
    # Preprocess
    transform = A.Compose([
        A.LongestMaxSize(max_size=256),  # Smaller size for crops
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    transformed = transform(image=crop_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = maskrcnn_model(input_tensor)[0]
    
    # Find best matching prediction
    masks = predictions['masks'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    # Map YOLO class to Mask R-CNN class
    target_class = YOLO_TO_MASKRCNN_CLASS[class_id]
    
    # Find highest scoring mask of the target class
    best_mask = None
    best_score = 0.0
    
    for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        if label == target_class and score > best_score:
            best_mask = mask[0]
            best_score = score
    
    # If no matching class found, use highest scoring mask regardless of class
    if best_mask is None and len(masks) > 0:
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx][0]
        best_score = scores[best_idx]
    
    # If still no mask, generate a simple ellipse mask
    if best_mask is None:
        h, w = crop.shape[:2]
        best_mask = np.zeros((256, 256), dtype=np.float32)
        center = (128, 128)
        axes = (100, 60)  # Ellipse axes
        cv2.ellipse(best_mask, center, axes, 0, 0, 360, 1.0, -1)
        best_score = 0.5  # Default confidence
    
    # Binarize mask
    mask_binary = (best_mask > mask_threshold).astype(np.uint8)
    
    # Resize mask back to crop size
    mask_resized = cv2.resize(mask_binary, (crop.shape[1], crop.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    return mask_resized, best_score


def generate_ellipse_mask(box: list, image_shape: tuple):
    """Generate an ellipse mask from bounding box (fallback)."""
    x1, y1, x2, y2 = box
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
    
    if axes[0] > 0 and axes[1] > 0:
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    
    return mask


# ============================================================================
# Pipeline
# ============================================================================

def run_pipeline(image_path: str, 
                 yolo_model_path: str = 'experiments/microplastic_yolo/weights/best.pt',
                 maskrcnn_model_path: str = 'experiments/maskrcnn_best.pth',
                 output_dir: str = 'experiments/pipeline_output',
                 yolo_conf: float = 0.25,
                 mask_threshold: float = 0.5,
                 use_maskrcnn: bool = True):
    """
    Run the complete YOLO + Mask R-CNN pipeline.
    
    Args:
        image_path: Path to input image
        yolo_model_path: Path to YOLO model
        maskrcnn_model_path: Path to Mask R-CNN model
        output_dir: Output directory for results
        yolo_conf: YOLO confidence threshold
        mask_threshold: Mask binarization threshold
        use_maskrcnn: If False, use ellipse masks from bboxes (faster)
    
    Returns:
        Dictionary with detections and masks
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print("YOLO + MASK R-CNN PIPELINE")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print(f"YOLO model: {yolo_model_path}")
    print(f"Mask R-CNN model: {maskrcnn_model_path}")
    print(f"YOLO confidence: {yolo_conf}")
    print(f"Use Mask R-CNN: {use_maskrcnn}")
    print(f"{'='*60}\n")
    
    # Load models
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_model_path)
    
    if use_maskrcnn:
        print("Loading Mask R-CNN model...")
        maskrcnn_model = load_maskrcnn_model(maskrcnn_model_path, device)
    else:
        maskrcnn_model = None
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Stage 1: YOLO Detection
    print("\n[Stage 1] Running YOLO detection...")
    detections = run_yolo_detection(yolo_model, image_path, yolo_conf)
    print(f"Detected {len(detections)} objects")
    
    # Stage 2: Mask R-CNN Segmentation (per crop)
    print(f"\n[Stage 2] Running segmentation on {len(detections)} crops...")
    
    # Create full-image mask overlay
    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    results = []
    for i, det in enumerate(detections):
        box = det['box']
        class_id = det['class_id']
        class_name = det['class_name']
        yolo_conf_score = det['confidence']
        
        if use_maskrcnn and maskrcnn_model is not None:
            # Crop and segment
            crop, crop_box, rel_box = crop_detection(image, box, padding=30)
            mask_crop, mask_conf = segment_crop(maskrcnn_model, crop, class_id, device, mask_threshold)
            
            # Place mask in original image coordinates
            x1_pad, y1_pad, x2_pad, y2_pad = crop_box
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y1_pad:y2_pad, x1_pad:x2_pad] = mask_crop
        else:
            # Fallback: ellipse mask from bounding box
            full_mask = generate_ellipse_mask(box, image.shape)
            mask_conf = 0.5
        
        # Apply color to mask overlay
        color = COLORS.get(YOLO_TO_MASKRCNN_CLASS[class_id], (255, 255, 255))
        mask_overlay[full_mask == 1] = color
        
        results.append({
            'box': box,
            'class_id': class_id,
            'class_name': class_name,
            'yolo_confidence': yolo_conf_score,
            'mask_confidence': mask_conf,
            'mask': full_mask
        })
        
        print(f"  [{i+1}] {class_name}: YOLO={yolo_conf_score:.2f}, Mask={mask_conf:.2f}")
    
    # Create visualization
    print("\n[Stage 3] Creating visualization...")
    
    vis_image = image.copy()
    
    # Blend mask overlay
    vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)
    
    # Draw bounding boxes and labels
    for det in results:
        box = det['box']
        class_name = det['class_name']
        yolo_conf_score = det['yolo_confidence']
        class_id = det['class_id']
        
        color = COLORS.get(YOLO_TO_MASKRCNN_CLASS[class_id], (255, 255, 255))
        x1, y1, x2, y2 = box
        
        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {yolo_conf_score:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_image, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)
        cv2.putText(vis_image, label, (x1 + 2, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{Path(image_path).stem}_pipeline.png"
    cv2.imwrite(str(output_path), vis_image)
    print(f"\nResult saved to: {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Total detections: {len(results)}")
    for name in ['fiber', 'film', 'fragment']:
        count = sum(1 for r in results if r['class_name'] == name)
        if count > 0:
            print(f"  - {name}: {count}")
    print(f"{'='*60}\n")
    
    return {
        'detections': results,
        'visualization_path': str(output_path)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='YOLO + Mask R-CNN Pipeline')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--yolo', type=str, default='experiments/microplastic_yolo/weights/best.pt',
                        help='YOLO model path')
    parser.add_argument('--maskrcnn', type=str, default='experiments/maskrcnn_best.pth',
                        help='Mask R-CNN model path')
    parser.add_argument('--output', type=str, default='experiments/pipeline_output',
                        help='Output directory')
    parser.add_argument('--yolo-conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--mask-threshold', type=float, default=0.5, help='Mask binarization threshold')
    parser.add_argument('--no-maskrcnn', action='store_true', 
                        help='Skip Mask R-CNN, use ellipse masks from bboxes')
    
    args = parser.parse_args()
    
    run_pipeline(
        image_path=args.image,
        yolo_model_path=args.yolo,
        maskrcnn_model_path=args.maskrcnn,
        output_dir=args.output,
        yolo_conf=args.yolo_conf,
        mask_threshold=args.mask_threshold,
        use_maskrcnn=not args.no_maskrcnn
    )


if __name__ == "__main__":
    main()
