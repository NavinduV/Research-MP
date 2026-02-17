"""
Cellpose Auto-Annotation for Microplastic Crops.

Alternative to SAM-based mask generation. Uses Cellpose's pretrained models
to automatically segment microplastic objects in cropped images. Cellpose
excels at segmenting objects in microscopy-like images without requiring
point/box prompts — it discovers objects automatically via gradient flow fields.

This is particularly useful for:
  - Elongated fibers (Cellpose handles elongated shapes well)
  - Irregular fragments and films
  - Batch processing without manual prompts

Output is fully compatible with the SAM pipeline output format so that
downstream training (Mask R-CNN, etc.) can consume either interchangeably.

Usage:
    # Generate masks for all crops (default: cyto3 model)
    python src/cellpose_auto_annotate.py --mode annotate --crops-dir data/crops

    # Use a specific model and custom diameter
    python src/cellpose_auto_annotate.py --mode annotate --crops-dir data/crops --model-type cyto3 --diameter 60

    # Annotate augmented crops
    python src/cellpose_auto_annotate.py --mode annotate --crops-dir data/crops_augmented

    # Visualize generated masks
    python src/cellpose_auto_annotate.py --mode visualize --crops-dir data/crops_cellpose --num-samples 20

    # Visualize a specific image
    python src/cellpose_auto_annotate.py --mode visualize --image data/crops_cellpose/images/sample.png

    # Convert to COCO format
    python src/cellpose_auto_annotate.py --mode convert --crops-dir data/crops_cellpose

    # Compare Cellpose vs SAM masks side-by-side
    python src/cellpose_auto_annotate.py --mode compare --crops-dir data/crops_cellpose --sam-dir data/crops_sam --num-samples 20

Requirements:
    pip install cellpose opencv-python
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

# Cellpose imports
try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("WARNING: cellpose not installed. Run: pip install cellpose")


# ============================================================================
# Configuration
# ============================================================================

CLASS_NAMES = ['background', 'fiber', 'film', 'fragment']
YOLO_TO_MASKRCNN = {0: 1, 1: 2, 2: 3}  # fiber=0->1, film=1->2, fragment=2->3

# Cellpose model options
# In cellpose v4+ only 'cpsam' is natively available. Legacy names (cyto3,
# cyto2, etc.) can still be passed as pretrained_model and will be downloaded
# from the cellpose model zoo on first use.
CELLPOSE_MODELS = {
    'cpsam':  'Cellpose-SAM hybrid — default v4+ model, best overall accuracy',
    'cyto3':  'Legacy cytoplasm model — good for diverse cell-like shapes',
    'cyto2':  'Legacy improved cytoplasm model — good generalisation',
    'cyto':   'Legacy general cytoplasm model — fast baseline',
    'nuclei': 'Legacy nuclear model — compact round/oval objects (fragments)',
}

# Per-class tuning suggestions (used as defaults when --class-aware is set)
CLASS_PARAMS = {
    'fiber': {
        'flow_threshold': 0.6,      # More lenient for elongated shapes
        'cellprob_threshold': -2.0,  # Lower threshold to capture thin fibers
        'min_size': 50,              # Fibers can be thin but long
    },
    'film': {
        'flow_threshold': 0.4,
        'cellprob_threshold': -1.0,
        'min_size': 100,
    },
    'fragment': {
        'flow_threshold': 0.4,
        'cellprob_threshold': 0.0,
        'min_size': 80,
    },
}


# ============================================================================
# Cellpose Mask Generation
# ============================================================================

def load_cellpose_model(model_type: str = 'cpsam', gpu: bool = True):
    """
    Load a Cellpose model (compatible with cellpose v4+).

    Args:
        model_type: Pretrained model name. In v4+, 'cpsam' is the native
                    default. Legacy names ('cyto3', 'cyto2', 'cyto', 'nuclei')
                    are still accepted and will be downloaded on first use.
        gpu: Whether to use GPU acceleration

    Returns:
        CellposeModel instance
    """
    if not CELLPOSE_AVAILABLE:
        raise ImportError("cellpose not installed. Run: pip install cellpose")

    use_gpu = gpu and torch.cuda.is_available()
    print(f"Loading Cellpose model: {model_type}")
    print(f"  GPU: {'enabled' if use_gpu else 'disabled (CPU mode)'}")
    if model_type in CELLPOSE_MODELS:
        print(f"  Description: {CELLPOSE_MODELS[model_type]}")

    # cellpose v4+ uses CellposeModel with pretrained_model= argument.
    # The old models.Cellpose class and model_type= kwarg no longer exist.
    model = models.CellposeModel(pretrained_model=model_type, gpu=use_gpu)
    return model


def generate_mask_with_cellpose(
    model,
    image: np.ndarray,
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
) -> Tuple[np.ndarray, float]:
    """
    Generate segmentation mask using Cellpose for a single crop image.

    Since each crop contains one centred microplastic object, we run Cellpose
    and then pick the mask whose centroid is closest to the image centre.

    Args:
        model: Cellpose model instance
        image: Image array (H x W x C), BGR or RGB
        diameter: Expected object diameter in pixels (None = auto-estimate)
        flow_threshold: Flow error threshold (higher = more lenient)
        cellprob_threshold: Cell probability threshold (lower = more pixels accepted)
        min_size: Minimum mask area in pixels

    Returns:
        mask: Binary mask (H x W) with values 0/1
        score: Confidence-like metric (mean flow probability inside the mask)
    """
    h, w = image.shape[:2]

    # Convert BGR -> RGB if needed (Cellpose expects RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    # Run Cellpose evaluation.
    # cellpose v4+ returns (masks, flows, styles) — no diams return value.
    result = model.eval(
        image_rgb,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    # Unpack flexibly to support both v3 (4 values) and v4+ (3 values)
    masks_out, flows = result[0], result[1]

    # masks_out is a label image: 0 = background, 1..N = object labels
    num_objects = masks_out.max()

    if num_objects == 0:
        # Cellpose found nothing — return an elliptical fallback mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2),
                     (int(w * 0.35), int(h * 0.35)), 0, 0, 360, 1, -1)
        return mask, 0.0

    # Pick the label whose centroid is closest to the image centre
    centre = np.array([h / 2, w / 2])
    best_label = 1
    best_dist = float('inf')

    for label_id in range(1, num_objects + 1):
        ys, xs = np.where(masks_out == label_id)
        if len(ys) == 0:
            continue
        centroid = np.array([ys.mean(), xs.mean()])
        dist = np.linalg.norm(centroid - centre)
        if dist < best_dist:
            best_dist = dist
            best_label = label_id

    binary_mask = (masks_out == best_label).astype(np.uint8)

    # Compute a confidence-like score from the cell probability map.
    # flows[2] is the cell probability map (H, W) float32.
    # In v4+ values are already in [0, 1] range; in v3 they were logits.
    if flows is not None and len(flows) > 2 and flows[2] is not None:
        cellprob = flows[2]  # shape (H, W)
        mask_pixels = cellprob[binary_mask == 1]
        if len(mask_pixels) > 0:
            raw_mean = float(mask_pixels.mean())
            # Normalise to [0,1]: if values are already in [0,1] keep them;
            # if they look like logits (outside [0,1]) apply sigmoid.
            if 0.0 <= raw_mean <= 1.0:
                score = raw_mean
            else:
                score = float(1.0 / (1.0 + np.exp(-np.clip(raw_mean, -20, 20))))
        else:
            score = 0.0
    else:
        score = float(binary_mask.sum()) / float(h * w)  # area ratio fallback

    return binary_mask, score


# ============================================================================
# Main Annotation Function
# ============================================================================

def annotate_crops(
    crops_dir: str,
    model_type: str = 'cyto3',
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    output_dir: str = None,
    class_aware: bool = False,
    gpu: bool = True,
):
    """
    Generate Cellpose masks for all crop images.

    Args:
        crops_dir: Directory containing crops and annotations.json
        model_type: Cellpose model type ('cyto', 'cyto2', 'cyto3', 'nuclei')
        diameter: Expected object diameter (None = auto-estimate per image)
        flow_threshold: Flow error threshold
        cellprob_threshold: Cell probability threshold
        min_size: Minimum mask area in pixels
        output_dir: Output directory (default: crops_dir with _cellpose suffix)
        class_aware: Use per-class tuned parameters from CLASS_PARAMS
        gpu: Use GPU if available
    """
    print(f"\n{'=' * 60}")
    print("CELLPOSE AUTO-ANNOTATION FOR MICROPLASTIC CROPS")
    print(f"{'=' * 60}")

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
        output_path = crops_path.parent / f"{crops_path.name}_cellpose"
    else:
        output_path = Path(output_dir)

    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'masks').mkdir(parents=True, exist_ok=True)

    # Load Cellpose model (default: cpsam for cellpose v4+)
    model = load_cellpose_model(model_type, gpu=gpu)

    # Process each crop
    new_annotations: Dict[str, dict] = {}
    images_dir = crops_path / 'images'

    success_count = 0
    fail_count = 0
    scores_by_class: Dict[str, List[float]] = {'fiber': [], 'film': [], 'fragment': []}

    for sample_name, ann in tqdm(annotations.items(), desc="Generating Cellpose masks"):
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

        try:
            # Determine per-image parameters
            class_name = ann.get('class_name', 'fragment')
            ft = flow_threshold
            cpt = cellprob_threshold
            ms = min_size

            if class_aware and class_name in CLASS_PARAMS:
                params = CLASS_PARAMS[class_name]
                ft = params.get('flow_threshold', ft)
                cpt = params.get('cellprob_threshold', cpt)
                ms = params.get('min_size', ms)

            mask, score = generate_mask_with_cellpose(
                model,
                image,
                diameter=diameter,
                flow_threshold=ft,
                cellprob_threshold=cpt,
                min_size=ms,
            )

            # Save mask (same format as SAM: 0/255 single-channel PNG)
            mask_filename = sample_name.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
            cv2.imwrite(str(output_path / 'masks' / mask_filename), mask * 255)

            # Copy original image
            cv2.imwrite(str(output_path / 'images' / sample_name), image)

            # Update annotation (compatible with SAM annotation format)
            new_ann = ann.copy()
            new_ann['mask_file'] = mask_filename
            new_ann['cellpose_score'] = score
            new_ann['mask_method'] = f'cellpose_{model_type}'
            new_ann['cellpose_params'] = {
                'model_type': model_type,
                'diameter': diameter,
                'flow_threshold': ft,
                'cellprob_threshold': cpt,
                'min_size': ms,
                'class_aware': class_aware,
            }
            new_annotations[sample_name] = new_ann

            if class_name in scores_by_class:
                scores_by_class[class_name].append(score)

            success_count += 1

        except Exception as e:
            print(f"  Error processing {sample_name}: {e}")
            fail_count += 1
            continue

    # Save new annotations
    with open(output_path / 'annotations.json', 'w') as f:
        json.dump(new_annotations, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("CELLPOSE ANNOTATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Model: {model_type}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"\nScore summary by class:")
    for cls, scores in scores_by_class.items():
        if scores:
            print(f"  {cls:>10s}: mean={np.mean(scores):.3f}  "
                  f"min={min(scores):.3f}  max={max(scores):.3f}  n={len(scores)}")
    print(f"\nOutput directory: {output_path}")
    print(f"  - Images: {output_path / 'images'}")
    print(f"  - Masks:  {output_path / 'masks'}")
    print(f"  - Annotations: {output_path / 'annotations.json'}")
    print(f"{'=' * 60}\n")

    return str(output_path)


# ============================================================================
# Visualization
# ============================================================================

def visualize_masks(
    crops_dir: str,
    num_samples: int = 20,
    specific_image: str = None,
    save_dir: str = None,
):
    """
    Visualize generated Cellpose masks.

    Args:
        crops_dir: Directory containing Cellpose-annotated crops
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

        # Panel 2: Mask (coloured)
        mask_colored = cv2.applyColorMap((mask_binary * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis[:, w:w * 2] = mask_colored

        # Panel 3: Overlay
        overlay = image.copy()
        overlay[mask_binary == 1] = (overlay[mask_binary == 1] * 0.5 +
                                      np.array([0, 200, 255]) * 0.5).astype(np.uint8)
        vis[:, w * 2:] = overlay

        # Add text
        class_name = ann.get('class_name', 'unknown')
        score = ann.get('cellpose_score', ann.get('sam_score', 0))
        method = ann.get('mask_method', 'unknown')
        cv2.putText(vis, f"Class: {class_name}", (10, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Score: {score:.3f}", (10, 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Method: {method}", (10, 60),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(vis, "Original", (10, h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, "Mask", (w + 10, h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis, "Overlay", (w * 2 + 10, h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if save_dir:
            cv2.imwrite(str(save_path / f"vis_{sample_name}"), vis)
            print(f"Saved: vis_{sample_name}")
        else:
            cv2.imshow(f"Cellpose Mask - {sample_name}", vis)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break

    if not save_dir:
        cv2.destroyAllWindows()


# ============================================================================
# Compare Cellpose vs SAM masks
# ============================================================================

def compare_masks(
    cellpose_dir: str,
    sam_dir: str,
    num_samples: int = 20,
    save_dir: str = None,
):
    """
    Side-by-side comparison of Cellpose and SAM masks for the same images.

    Args:
        cellpose_dir: Directory with Cellpose-annotated crops
        sam_dir: Directory with SAM-annotated crops
        num_samples: Number of samples to compare
        save_dir: Directory to save comparison images
    """
    cellpose_path = Path(cellpose_dir)
    sam_path = Path(sam_dir)

    with open(cellpose_path / 'annotations.json') as f:
        cellpose_ann = json.load(f)
    with open(sam_path / 'annotations.json') as f:
        sam_ann = json.load(f)

    # Find common images
    common = sorted(set(cellpose_ann.keys()) & set(sam_ann.keys()))
    if not common:
        print("No common images found between Cellpose and SAM directories.")
        return

    import random
    samples = random.sample(common, min(num_samples, len(common)))

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    iou_scores = []

    for sample_name in samples:
        c_ann = cellpose_ann[sample_name]
        s_ann = sam_ann[sample_name]

        # Load image
        img_path = cellpose_path / 'images' / sample_name
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]

        # Load Cellpose mask
        c_mask_file = c_ann.get('mask_file', sample_name.replace('.png', '_mask.png'))
        c_mask_path = cellpose_path / 'masks' / c_mask_file
        if c_mask_path.exists():
            c_mask = cv2.imread(str(c_mask_path), cv2.IMREAD_GRAYSCALE)
            c_mask_bin = (c_mask > 127).astype(np.uint8)
        else:
            c_mask_bin = np.zeros((h, w), dtype=np.uint8)

        # Load SAM mask
        s_mask_file = s_ann.get('mask_file', sample_name.replace('.png', '_mask.png'))
        s_mask_path = sam_path / 'masks' / s_mask_file
        if s_mask_path.exists():
            s_mask = cv2.imread(str(s_mask_path), cv2.IMREAD_GRAYSCALE)
            s_mask_bin = (s_mask > 127).astype(np.uint8)
        else:
            s_mask_bin = np.zeros((h, w), dtype=np.uint8)

        # Compute IoU
        intersection = np.logical_and(c_mask_bin, s_mask_bin).sum()
        union = np.logical_or(c_mask_bin, s_mask_bin).sum()
        iou = float(intersection) / float(union) if union > 0 else 0.0
        iou_scores.append(iou)

        # Build comparison: Original | SAM | Cellpose | Diff
        vis = np.zeros((h, w * 4, 3), dtype=np.uint8)

        # Panel 1: Original
        vis[:, :w] = image

        # Panel 2: SAM overlay (green)
        sam_overlay = image.copy()
        sam_overlay[s_mask_bin == 1] = (sam_overlay[s_mask_bin == 1] * 0.5 +
                                         np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        vis[:, w:w * 2] = sam_overlay

        # Panel 3: Cellpose overlay (orange)
        cp_overlay = image.copy()
        cp_overlay[c_mask_bin == 1] = (cp_overlay[c_mask_bin == 1] * 0.5 +
                                        np.array([0, 165, 255]) * 0.5).astype(np.uint8)
        vis[:, w * 2:w * 3] = cp_overlay

        # Panel 4: Difference map
        diff = np.zeros((h, w, 3), dtype=np.uint8)
        both = np.logical_and(c_mask_bin, s_mask_bin)       # white  — agreement
        only_sam = np.logical_and(s_mask_bin, ~c_mask_bin.astype(bool))   # green  — SAM only
        only_cp = np.logical_and(c_mask_bin, ~s_mask_bin.astype(bool))    # orange — Cellpose only
        diff[both] = [255, 255, 255]
        diff[only_sam] = [0, 255, 0]
        diff[only_cp] = [0, 165, 255]
        vis[:, w * 3:] = diff

        # Labels
        class_name = c_ann.get('class_name', 'unknown')
        c_score = c_ann.get('cellpose_score', 0)
        s_score = s_ann.get('sam_score', 0)

        cv2.putText(vis, f"{class_name} | IoU={iou:.2f}", (10, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(vis, "Original", (10, h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(vis, f"SAM ({s_score:.2f})", (w + 10, h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(vis, f"Cellpose ({c_score:.2f})", (w * 2 + 10, h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
        cv2.putText(vis, "Diff", (w * 3 + 10, h - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        if save_dir:
            cv2.imwrite(str(save_path / f"cmp_{sample_name}"), vis)
            print(f"Saved: cmp_{sample_name}")
        else:
            cv2.imshow(f"Compare - {sample_name}", vis)
            key = cv2.waitKey(0)
            if key == 27:
                break

    if not save_dir:
        cv2.destroyAllWindows()

    # Print IoU summary
    if iou_scores:
        print(f"\n{'=' * 40}")
        print("CELLPOSE vs SAM — IoU SUMMARY")
        print(f"{'=' * 40}")
        print(f"  Samples compared: {len(iou_scores)}")
        print(f"  Mean IoU:  {np.mean(iou_scores):.3f}")
        print(f"  Median IoU: {np.median(iou_scores):.3f}")
        print(f"  Min IoU:   {min(iou_scores):.3f}")
        print(f"  Max IoU:   {max(iou_scores):.3f}")
        print(f"{'=' * 40}")


# ============================================================================
# Convert to COCO Format
# ============================================================================

def convert_to_coco_format(crops_dir: str, output_file: str = None):
    """
    Convert Cellpose annotations to COCO format for compatibility with other
    tools (identical output format to SAM converter).

    Args:
        crops_dir: Directory containing Cellpose-annotated crops
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
            {"id": 3, "name": "fragment"},
        ],
    }

    ann_id = 1
    for img_id, (sample_name, ann) in enumerate(annotations.items(), start=1):
        img_path = crops_path / 'images' / sample_name
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": sample_name,
            "width": w,
            "height": h,
        })

        mask_file = ann.get('mask_file')
        if mask_file:
            mask_path = crops_path / 'masks' / mask_file
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask_binary = (mask > 127).astype(np.uint8)

                # Convert to RLE
                rle = mask_utils.encode(np.asfortranarray(mask_binary))
                rle['counts'] = rle['counts'].decode('utf-8')

                # Bounding box
                contours, _ = cv2.findContours(
                    mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                    "iscrowd": 0,
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
    parser = argparse.ArgumentParser(
        description='Cellpose Auto-Annotation for Microplastic Crops')

    parser.add_argument(
        '--mode', type=str,
        choices=['annotate', 'visualize', 'convert', 'compare'],
        default='annotate',
        help='Mode: annotate, visualize, convert (COCO), or compare (vs SAM)')

    # Input / output
    parser.add_argument(
        '--crops-dir', type=str, default='data/crops',
        help='Directory containing crop images and annotations')
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for Cellpose annotations')

    # Cellpose model parameters
    parser.add_argument(
        '--model-type', type=str,
        choices=list(CELLPOSE_MODELS.keys()),
        default='cpsam',
        help='Cellpose model type (default: cpsam — native v4+ model)')
    parser.add_argument(
        '--diameter', type=float, default=None,
        help='Expected object diameter in pixels (None = auto-estimate)')
    parser.add_argument(
        '--flow-threshold', type=float, default=0.4,
        help='Flow error threshold (default: 0.4, higher = more lenient)')
    parser.add_argument(
        '--cellprob-threshold', type=float, default=0.0,
        help='Cell probability threshold (default: 0.0, lower = more pixels)')
    parser.add_argument(
        '--min-size', type=int, default=15,
        help='Minimum mask area in pixels (default: 15)')
    parser.add_argument(
        '--class-aware', action='store_true',
        help='Use per-class tuned parameters (fiber/film/fragment)')

    # Visualization / comparison
    parser.add_argument(
        '--num-samples', type=int, default=20,
        help='Number of samples to visualize or compare')
    parser.add_argument(
        '--image', type=str, default=None,
        help='Specific image to visualize')
    parser.add_argument(
        '--save-vis', type=str, default=None,
        help='Directory to save visualizations')
    parser.add_argument(
        '--sam-dir', type=str, default=None,
        help='SAM annotations directory (for compare mode)')

    # Device
    parser.add_argument(
        '--no-gpu', action='store_true',
        help='Disable GPU, force CPU mode')

    args = parser.parse_args()

    if args.mode == 'annotate':
        output_path = annotate_crops(
            crops_dir=args.crops_dir,
            model_type=args.model_type,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
            output_dir=args.output_dir,
            class_aware=args.class_aware,
            gpu=not args.no_gpu,
        )
        print(f"\nNext steps:")
        print(f"  1. Visualize masks:")
        print(f"     python src/cellpose_auto_annotate.py --mode visualize "
              f"--crops-dir {output_path}")
        print(f"  2. Compare with SAM masks:")
        print(f"     python src/cellpose_auto_annotate.py --mode compare "
              f"--crops-dir {output_path} --sam-dir data/crops_sam")
        print(f"  3. Convert to COCO format:")
        print(f"     python src/cellpose_auto_annotate.py --mode convert "
              f"--crops-dir {output_path}")

    elif args.mode == 'visualize':
        visualize_masks(
            crops_dir=args.crops_dir,
            num_samples=args.num_samples,
            specific_image=args.image,
            save_dir=args.save_vis,
        )

    elif args.mode == 'convert':
        convert_to_coco_format(crops_dir=args.crops_dir)

    elif args.mode == 'compare':
        sam_dir = args.sam_dir
        if sam_dir is None:
            # Try to guess SAM directory from the crops dir name
            crops_path = Path(args.crops_dir)
            sam_guess = crops_path.parent / crops_path.name.replace('_cellpose', '_sam')
            if sam_guess.exists():
                sam_dir = str(sam_guess)
            else:
                print("ERROR: --sam-dir is required for compare mode.")
                print(f"  Tried: {sam_guess} (does not exist)")
                return

        compare_masks(
            cellpose_dir=args.crops_dir,
            sam_dir=sam_dir,
            num_samples=args.num_samples,
            save_dir=args.save_vis,
        )


if __name__ == "__main__":
    main()
