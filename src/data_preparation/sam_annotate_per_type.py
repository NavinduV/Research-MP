"""
SAM Auto-Annotation Per Microplastic Type.

Generates SAM segmentation masks separately for each microplastic type
(fiber, film, fragment) from the unified crops directory.  Each type gets
its own output folder under data/:

    data/crops_fiber_sam/   images/ masks/ annotations.json
    data/crops_film_sam/    images/ masks/ annotations.json
    data/crops_fragment_sam/ images/ masks/ annotations.json

This prepares type-specific datasets for training 3 specialised Mask R-CNN
models (one per type).

Usage:
    # Annotate all three types (default)
    python src/data_preparation/sam_annotate_per_type.py

    # Annotate only fiber crops
    python src/data_preparation/sam_annotate_per_type.py --types fiber

    # Use a specific SAM model
    python src/data_preparation/sam_annotate_per_type.py --sam-model vit_h

    # Visualise generated masks for a type
    python src/data_preparation/sam_annotate_per_type.py --mode visualize --types fragment --num-samples 20
"""

import argparse
import json
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple

# Re-use helpers from the existing SAM annotator
try:
    from src.data_preparation.sam_auto_annotate import (
        load_sam_model,
        generate_mask_with_sam,
        generate_masks_automatic,
        visualize_masks,
        SAM_AVAILABLE,
    )
except ImportError:
    from sam_auto_annotate import (
        load_sam_model,
        generate_mask_with_sam,
        generate_masks_automatic,
        visualize_masks,
        SAM_AVAILABLE,
    )

# ============================================================================
# Configuration
# ============================================================================

TYPES = ['fiber', 'film', 'fragment']
YOLO_CLASS_IDS = {'fiber': 0, 'film': 1, 'fragment': 2}


# ============================================================================
# Helpers
# ============================================================================

def _split_annotations_by_type(annotations: dict) -> Dict[str, dict]:
    """Partition a single annotations dict into one dict per class_name."""
    per_type: Dict[str, dict] = {t: {} for t in TYPES}
    for fname, ann in annotations.items():
        cls = ann.get('class_name', '')
        if cls in per_type:
            per_type[cls][fname] = ann
    return per_type


def _resolve_image_path(crops_path: Path, sample_name: str, ann: dict) -> Path:
    """Find the image file — check images/ first, then class subdir."""
    flat = crops_path / 'images' / sample_name
    if flat.exists():
        return flat
    cls_name = ann.get('class_name', '')
    subdir = crops_path / cls_name / sample_name
    if subdir.exists():
        return subdir
    return flat  # caller will handle missing file


# ============================================================================
# Per-Type Annotation
# ============================================================================

def annotate_per_type(
    crops_dir: str = 'data/crops',
    types: Optional[List[str]] = None,
    sam_model_type: str = 'vit_h',
    sam_checkpoint: str = None,
    output_root: str = 'data',
    use_automatic: bool = False,
    device: str = None,
):
    """
    Generate SAM masks for each microplastic type independently.

    Creates output directories:  <output_root>/crops_<type>_sam/
    """
    if types is None:
        types = TYPES

    crops_path = Path(crops_dir)
    output_root_path = Path(output_root)

    # Load annotations
    ann_file = crops_path / 'annotations.json'
    if not ann_file.exists():
        raise FileNotFoundError(f"annotations.json not found in {crops_dir}")

    with open(ann_file) as f:
        all_annotations = json.load(f)

    per_type = _split_annotations_by_type(all_annotations)

    # Print summary
    print(f"\n{'='*60}")
    print("SAM PER-TYPE ANNOTATION FOR MICROPLASTIC CROPS")
    print(f"{'='*60}")
    for t in TYPES:
        print(f"  {t:>10}: {len(per_type[t])} crops")
    print(f"  Types to process: {types}")
    print(f"{'='*60}\n")

    # Load SAM once — shared across all types
    sam, device = load_sam_model(sam_model_type, sam_checkpoint, device)

    if use_automatic:
        from segment_anything import SamAutomaticMaskGenerator
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
        predictor = None
    else:
        from segment_anything import SamPredictor
        predictor = SamPredictor(sam)
        mask_generator = None

    # Process each requested type
    for mp_type in types:
        annotations = per_type.get(mp_type, {})
        if not annotations:
            print(f"\n[{mp_type}] No crops found — skipping.")
            continue

        out_path = output_root_path / f'crops_{mp_type}_sam'
        (out_path / 'images').mkdir(parents=True, exist_ok=True)
        (out_path / 'masks').mkdir(parents=True, exist_ok=True)

        new_annotations: dict = {}
        success = 0
        fail = 0

        print(f"\n{'='*60}")
        print(f"Processing {mp_type.upper()} ({len(annotations)} crops)")
        print(f"Output: {out_path}")
        print(f"{'='*60}")

        for sample_name, ann in tqdm(annotations.items(), desc=f"SAM [{mp_type}]"):
            img_path = _resolve_image_path(crops_path, sample_name, ann)

            if not img_path.exists():
                print(f"  Warning: image not found: {img_path}")
                fail += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  Warning: could not read: {img_path}")
                fail += 1
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                if use_automatic:
                    mask, score = generate_masks_automatic(mask_generator, image_rgb)
                else:
                    predictor.set_image(image_rgb)
                    rel_box = ann.get('rel_box')
                    mask, score = generate_mask_with_sam(
                        predictor,
                        image_rgb,
                        use_center_point=True,
                        use_box_prompt=(rel_box is not None),
                        rel_box=rel_box,
                    )

                mask_filename = sample_name.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
                cv2.imwrite(str(out_path / 'masks' / mask_filename), mask * 255)
                cv2.imwrite(str(out_path / 'images' / sample_name), image)

                new_ann = ann.copy()
                new_ann['mask_file'] = mask_filename
                new_ann['sam_score'] = score
                new_ann['mask_method'] = 'sam_automatic' if use_automatic else 'sam_point_prompt'
                new_annotations[sample_name] = new_ann
                success += 1

            except Exception as e:
                print(f"  Error [{sample_name}]: {e}")
                fail += 1

        # Save type-specific annotations
        with open(out_path / 'annotations.json', 'w') as f:
            json.dump(new_annotations, f, indent=2)

        print(f"\n  {mp_type} done — success: {success}, failed: {fail}")
        print(f"  Output: {out_path}")

    print(f"\n{'='*60}")
    print("ALL PER-TYPE SAM ANNOTATIONS COMPLETE")
    print(f"{'='*60}\n")


# ============================================================================
# Visualise per-type masks
# ============================================================================

def visualize_per_type(
    types: Optional[List[str]] = None,
    output_root: str = 'data',
    num_samples: int = 20,
    save_dir: str = None,
):
    """Visualise SAM masks for one or more types."""
    if types is None:
        types = TYPES

    for mp_type in types:
        crops_dir = str(Path(output_root) / f'crops_{mp_type}_sam')
        if not Path(crops_dir).exists():
            print(f"[{mp_type}] Directory not found: {crops_dir} — skipping.")
            continue

        print(f"\n--- Visualising {mp_type} from {crops_dir} ---")
        vis_save = None
        if save_dir:
            vis_save = str(Path(save_dir) / mp_type)

        visualize_masks(
            crops_dir=crops_dir,
            num_samples=num_samples,
            save_dir=vis_save,
        )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SAM Per-Type Annotation for Microplastic Crops')
    parser.add_argument('--mode', choices=['annotate', 'visualize'], default='annotate',
                        help='annotate: generate masks; visualize: view masks')
    parser.add_argument('--crops-dir', default='data/crops',
                        help='Source crops directory with annotations.json')
    parser.add_argument('--output-root', default='data',
                        help='Root directory for per-type output folders')
    parser.add_argument('--types', nargs='+', default=None,
                        choices=TYPES,
                        help='Which types to process (default: all)')
    parser.add_argument('--sam-model', choices=['vit_h', 'vit_l', 'vit_b'], default='vit_h',
                        help='SAM model type')
    parser.add_argument('--sam-checkpoint', default=None,
                        help='Path to SAM checkpoint')
    parser.add_argument('--automatic', action='store_true',
                        help='Use automatic mask generation instead of point prompts')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Samples to visualise per type')
    parser.add_argument('--save-vis', default=None,
                        help='Directory to save visualisations')
    parser.add_argument('--device', default=None,
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    if args.mode == 'annotate':
        annotate_per_type(
            crops_dir=args.crops_dir,
            types=args.types,
            sam_model_type=args.sam_model,
            sam_checkpoint=args.sam_checkpoint,
            output_root=args.output_root,
            use_automatic=args.automatic,
            device=args.device,
        )
        print("\nNext steps:")
        print("  1. Visualise:  python src/data_preparation/sam_annotate_per_type.py --mode visualize")
        print("  2. Train:      python src/train/train_maskrcnn_per_type.py --types fiber film fragment")

    elif args.mode == 'visualize':
        visualize_per_type(
            types=args.types,
            output_root=args.output_root,
            num_samples=args.num_samples,
            save_dir=args.save_vis,
        )


if __name__ == '__main__':
    main()
