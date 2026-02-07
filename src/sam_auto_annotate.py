"""
SAM (Segment Anything Model) Auto-Annotation for Microplastic Crops.

This script uses Meta's SAM to automatically generate high-quality segmentation
masks for cropped microplastic images. These masks can then be used to train
Mask R-CNN for better segmentation accuracy.

Usage:
    # First time: Download SAM model checkpoint
    # From: https://github.com/facebookresearch/segment-anything#model-checkpoints
    # Place in project root (e.g., sam_vit_b_01ec64.pth)
    
    # Generate masks for all crops
    python src/sam_auto_annotate.py --mode annotate --crops-dir data/crops
    
    # Visualize generated masks
    python src/sam_auto_annotate.py --mode visualize --crops-dir data/crops --num-samples 20
    
    # Visualize specific image
    python src/sam_auto_annotate.py --mode visualize --image data/crops/images/sample.png

Requirements:
    pip install segment-anything opencv-python
"""

import argparse
import json
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("WARNING: segment-anything not installed. Run: pip install segment-anything")


# ============================================================================
# Configuration
# ============================================================================

CLASS_NAMES = ['background', 'fiber', 'film', 'fragment']
YOLO_TO_MASKRCNN = {0: 1, 1: 2, 2: 3}  # fiber=0->1, film=1->2, fragment=2->3

# SAM model options (download from: https://github.com/facebookresearch/segment-anything)
SAM_MODELS = {
    'vit_h': 'sam_vit_h_4b8939.pth',    # Largest, most accurate (~2.4GB)
    'vit_l': 'sam_vit_l_0b3195.pth',    # Medium (~1.2GB)
    'vit_b': 'sam_vit_b_01ec64.pth',    # Smallest, fastest (~375MB)
}


# ============================================================================
# SAM Mask Generation
# ============================================================================

def load_sam_model(model_type: str = 'vit_b', checkpoint_path: str = None, device: str = None):
    """
    Load SAM model.
    
    Args:
        model_type: One of 'vit_h', 'vit_l', 'vit_b'
        checkpoint_path: Path to checkpoint file (optional, will search common locations)
        device: Device to use ('cuda' or 'cpu')
    """
    if not SAM_AVAILABLE:
        raise ImportError("segment-anything not installed. Run: pip install segment-anything")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_name = SAM_MODELS.get(model_type)
        search_paths = [
            Path(checkpoint_name),
            Path('models') / checkpoint_name,
            Path.home() / '.cache' / 'sam' / checkpoint_name,
        ]
        
        for path in search_paths:
            if path.exists():
                checkpoint_path = str(path)
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"SAM checkpoint not found. Please download from:\n"
                f"https://github.com/facebookresearch/segment-anything#model-checkpoints\n"
                f"Searched locations: {[str(p) for p in search_paths]}"
            )
    
    print(f"Loading SAM model ({model_type}) from: {checkpoint_path}")
    print(f"Device: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()
    
    return sam, device


def generate_mask_with_sam(
    predictor: SamPredictor,
    image: np.ndarray,
    use_center_point: bool = True,
    use_box_prompt: bool = False,
    rel_box: Optional[List[int]] = None
) -> Tuple[np.ndarray, float]:
    """
    Generate segmentation mask using SAM.
    
    Args:
        predictor: SAM predictor with image already set
        image: Image array (HxWxC)
        use_center_point: Use center point as prompt
        use_box_prompt: Use bounding box as prompt
        rel_box: Relative box [x1, y1, x2, y2] within the crop
        
    Returns:
        mask: Binary mask (HxW)
        score: Confidence score
    """
    h, w = image.shape[:2]
    
    # Strategy 1: Use point prompt at center (most common for centered crops)
    if use_center_point:
        center_x, center_y = w // 2, h // 2
        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])  # 1 = foreground
        
        if use_box_prompt and rel_box is not None:
            # Combine point and box prompt
            input_box = np.array(rel_box)
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=True
            )
        else:
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
    elif rel_box is not None:
        # Strategy 2: Use box prompt only
        input_box = np.array(rel_box)
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=True
        )
    else:
        # Strategy 3: Use automatic mask generation for single best object
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
    
    # Select best mask (highest confidence)
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]
    
    return best_mask.astype(np.uint8), float(best_score)


def generate_masks_automatic(
    mask_generator: SamAutomaticMaskGenerator,
    image: np.ndarray,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.95
) -> Tuple[np.ndarray, float]:
    """
    Generate mask using automatic mode - finds all objects.
    
    Args:
        mask_generator: SAM automatic mask generator
        image: Image array (HxWxC)
        min_area_ratio: Minimum area as ratio of image size
        max_area_ratio: Maximum area as ratio of image size
        
    Returns:
        mask: Binary mask of the main object
        score: Stability score
    """
    h, w = image.shape[:2]
    image_area = h * w
    
    # Generate all masks
    masks_output = mask_generator.generate(image)
    
    if not masks_output:
        # Fallback: return center ellipse
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w//2, h//2), (int(w*0.35), int(h*0.35)), 0, 0, 360, 1, -1)
        return mask, 0.5
    
    # Filter masks by area and select best one
    valid_masks = []
    for mask_data in masks_output:
        area = mask_data['area']
        area_ratio = area / image_area
        
        if min_area_ratio <= area_ratio <= max_area_ratio:
            valid_masks.append(mask_data)
    
    if not valid_masks:
        # Pick the largest valid mask
        valid_masks = sorted(masks_output, key=lambda x: x['area'], reverse=True)
    
    # Select mask closest to center of image
    center = np.array([w // 2, h // 2])
    best_mask_data = min(valid_masks, key=lambda x: np.linalg.norm(
        np.array([x['bbox'][0] + x['bbox'][2]/2, x['bbox'][1] + x['bbox'][3]/2]) - center
    ))
    
    return best_mask_data['segmentation'].astype(np.uint8), best_mask_data['stability_score']


# ============================================================================
# Main Annotation Function
# ============================================================================

def annotate_crops(
    crops_dir: str,
    sam_model_type: str = 'vit_b',
    sam_checkpoint: str = None,
    output_dir: str = None,
    use_automatic: bool = False,
    device: str = None
):
    """
    Generate SAM masks for all crop images.
    
    Args:
        crops_dir: Directory containing crops and annotations.json
        sam_model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        sam_checkpoint: Path to SAM checkpoint
        output_dir: Output directory (default: crops_dir with _sam suffix)
        use_automatic: Use automatic mask generation instead of point prompts
        device: Device to use
    """
    print(f"\n{'='*60}")
    print("SAM AUTO-ANNOTATION FOR MICROPLASTIC CROPS")
    print(f"{'='*60}")
    
    crops_path = Path(crops_dir)
    if not crops_path.exists():
        raise FileNotFoundError(f"Crops directory not found: {crops_dir}")
    
    # Load existing annotations
    ann_file = crops_path / 'annotations.json'
    if not ann_file.exists():
        raise FileNotFoundError(f"annotations.json not found in {crops_dir}")
    
    with open(ann_file) as f:
        annotations = json.load(f)
    
    print(f"Found {len(annotations)} crop annotations")
    
    # Setup output directory
    if output_dir is None:
        output_path = crops_path.parent / f"{crops_path.name}_sam"
    else:
        output_path = Path(output_dir)
    
    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Load SAM model
    sam, device = load_sam_model(sam_model_type, sam_checkpoint, device)
    
    if use_automatic:
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100  # Filter tiny masks
        )
        predictor = None
    else:
        predictor = SamPredictor(sam)
        mask_generator = None
    
    # Process each crop
    new_annotations = {}
    images_dir = crops_path / 'images'
    
    success_count = 0
    fail_count = 0
    
    for sample_name, ann in tqdm(annotations.items(), desc="Generating SAM masks"):
        img_path = images_dir / sample_name
        
        if not img_path.exists():
            print(f"  Warning: Image not found: {img_path}")
            fail_count += 1
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Warning: Could not read: {img_path}")
            fail_count += 1
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            if use_automatic:
                # Automatic mask generation
                mask, score = generate_masks_automatic(mask_generator, image_rgb)
            else:
                # Point/box prompt based
                predictor.set_image(image_rgb)
                rel_box = ann.get('rel_box')
                mask, score = generate_mask_with_sam(
                    predictor,
                    image_rgb,
                    use_center_point=True,
                    use_box_prompt=(rel_box is not None),
                    rel_box=rel_box
                )
            
            # Save mask
            mask_filename = sample_name.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
            cv2.imwrite(str(output_path / 'masks' / mask_filename), mask * 255)
            
            # Copy original image
            cv2.imwrite(str(output_path / 'images' / sample_name), image)
            
            # Update annotation
            new_ann = ann.copy()
            new_ann['mask_file'] = mask_filename
            new_ann['sam_score'] = score
            new_ann['mask_method'] = 'sam_automatic' if use_automatic else 'sam_point_prompt'
            new_annotations[sample_name] = new_ann
            
            success_count += 1
            
        except Exception as e:
            print(f"  Error processing {sample_name}: {e}")
            fail_count += 1
            continue
    
    # Save new annotations
    with open(output_path / 'annotations.json', 'w') as f:
        json.dump(new_annotations, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SAM ANNOTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output directory: {output_path}")
    print(f"  - Images: {output_path / 'images'}")
    print(f"  - Masks: {output_path / 'masks'}")
    print(f"  - Annotations: {output_path / 'annotations.json'}")
    print(f"{'='*60}\n")
    
    return str(output_path)


# ============================================================================
# Visualization
# ============================================================================

def visualize_masks(
    crops_dir: str,
    num_samples: int = 20,
    specific_image: str = None,
    save_dir: str = None
):
    """
    Visualize generated SAM masks.
    
    Args:
        crops_dir: Directory containing SAM-annotated crops
        num_samples: Number of random samples to visualize
        specific_image: Specific image to visualize
        save_dir: Directory to save visualizations
    """
    crops_path = Path(crops_dir)
    
    # Load annotations
    ann_file = crops_path / 'annotations.json'
    with open(ann_file) as f:
        annotations = json.load(f)
    
    if specific_image:
        samples = [Path(specific_image).name]
    else:
        import random
        samples = random.sample(list(annotations.keys()), min(num_samples, len(annotations)))
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    for sample_name in samples:
        ann = annotations.get(sample_name)
        if ann is None:
            continue
        
        # Load image and mask
        img_path = crops_path / 'images' / sample_name
        mask_file = ann.get('mask_file')
        
        if mask_file:
            mask_path = crops_path / 'masks' / mask_file
        else:
            # Check if masks folder exists with default naming
            mask_path = crops_path / 'masks' / sample_name.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_binary = (mask > 127).astype(np.uint8)
        else:
            print(f"  Mask not found for {sample_name}")
            continue
        
        # Create visualization
        h, w = image.shape[:2]
        vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Panel 1: Original image
        vis[:, :w] = image
        
        # Panel 2: Mask
        mask_colored = cv2.applyColorMap((mask_binary * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis[:, w:w*2] = mask_colored
        
        # Panel 3: Overlay
        overlay = image.copy()
        overlay[mask_binary == 1] = overlay[mask_binary == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
        vis[:, w*2:] = overlay.astype(np.uint8)
        
        # Add text
        class_name = ann.get('class_name', 'unknown')
        score = ann.get('sam_score', 0)
        cv2.putText(vis, f"Class: {class_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"SAM Score: {score:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, "Original", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, "Mask", (w+10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, "Overlay", (w*2+10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if save_dir:
            cv2.imwrite(str(save_path / f"vis_{sample_name}"), vis)
            print(f"Saved: vis_{sample_name}")
        else:
            cv2.imshow(f"SAM Mask - {sample_name}", vis)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break
    
    if not save_dir:
        cv2.destroyAllWindows()


# ============================================================================
# Convert to COCO Format (for other tools)
# ============================================================================

def convert_to_coco_format(crops_dir: str, output_file: str = None):
    """
    Convert SAM annotations to COCO format for compatibility with other tools.
    
    Args:
        crops_dir: Directory containing SAM-annotated crops
        output_file: Output COCO JSON file
    """
    from pycocotools import mask as mask_utils
    
    crops_path = Path(crops_dir)
    
    with open(crops_path / 'annotations.json') as f:
        annotations = json.load(f)
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "fiber"},
            {"id": 2, "name": "film"},
            {"id": 3, "name": "fragment"}
        ]
    }
    
    ann_id = 1
    for img_id, (sample_name, ann) in enumerate(annotations.items(), start=1):
        # Load image dimensions
        img_path = crops_path / 'images' / sample_name
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        
        coco["images"].append({
            "id": img_id,
            "file_name": sample_name,
            "width": w,
            "height": h
        })
        
        # Load mask
        mask_file = ann.get('mask_file')
        if mask_file:
            mask_path = crops_path / 'masks' / mask_file
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask_binary = (mask > 127).astype(np.uint8)
                
                # Convert to RLE
                rle = mask_utils.encode(np.asfortranarray(mask_binary))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                # Get bounding box
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, bw, bh = cv2.boundingRect(contours[0])
                    area = cv2.contourArea(contours[0])
                else:
                    x, y, bw, bh = 0, 0, w, h
                    area = w * h
                
                class_id = YOLO_TO_MASKRCNN.get(ann.get('class_id', 0), 1)
                
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "segmentation": rle,
                    "area": float(area),
                    "bbox": [x, y, bw, bh],
                    "iscrowd": 0
                })
                ann_id += 1
    
    if output_file is None:
        output_file = str(crops_path / 'coco_annotations.json')
    
    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=2)
    
    print(f"COCO annotations saved to: {output_file}")
    return output_file


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SAM Auto-Annotation for Microplastic Crops')
    parser.add_argument('--mode', type=str, choices=['annotate', 'visualize', 'convert'],
                        default='annotate', help='Mode: annotate, visualize, or convert to COCO')
    parser.add_argument('--crops-dir', type=str, default='data/crops',
                        help='Directory containing crop images and annotations')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for SAM annotations')
    parser.add_argument('--sam-model', type=str, choices=['vit_h', 'vit_l', 'vit_b'],
                        default='vit_b', help='SAM model type')
    parser.add_argument('--sam-checkpoint', type=str, default=None,
                        help='Path to SAM checkpoint file')
    parser.add_argument('--automatic', action='store_true',
                        help='Use automatic mask generation instead of point prompts')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--image', type=str, default=None,
                        help='Specific image to visualize')
    parser.add_argument('--save-vis', type=str, default=None,
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    if args.mode == 'annotate':
        output_path = annotate_crops(
            crops_dir=args.crops_dir,
            sam_model_type=args.sam_model,
            sam_checkpoint=args.sam_checkpoint,
            output_dir=args.output_dir,
            use_automatic=args.automatic,
            device=args.device
        )
        print(f"\nNext steps:")
        print(f"1. Visualize masks: python src/sam_auto_annotate.py --mode visualize --crops-dir {output_path}")
        print(f"2. Update training to use SAM masks (see instructions below)")
        
    elif args.mode == 'visualize':
        visualize_masks(
            crops_dir=args.crops_dir,
            num_samples=args.num_samples,
            specific_image=args.image,
            save_dir=args.save_vis
        )
        
    elif args.mode == 'convert':
        convert_to_coco_format(crops_dir=args.crops_dir)


if __name__ == "__main__":
    main()
