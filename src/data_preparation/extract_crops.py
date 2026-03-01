"""
Extract crops from a YOLO-format dataset for EfficientNet training.

Reads YOLO images + labels (normalized xywh), crops each detection
with padding, and saves into {output}/{train,val}/{fiber,film,fragment}/.

Usage:
    python src/data_preparation/extract_crops.py \
        --yolo-dir data/yolo_augmented_balanced \
        --output   data/crops_yolo_balanced \
        --padding  15

    # Use a different class mapping (default: 0=fiber, 1=film, 2=fragment)
    python src/data_preparation/extract_crops.py \
        --yolo-dir data/yolo_augmented_balanced \
        --output   data/crops_yolo_balanced \
        --padding  20
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import cv2
import yaml
from tqdm import tqdm


# Default class mapping (matches dataset.yaml)
DEFAULT_CLASS_NAMES = {0: 'fiber', 1: 'film', 2: 'fragment'}


def load_class_names(yolo_dir: Path) -> dict:
    """Load class names from dataset.yaml, falling back to defaults."""
    yaml_path = yolo_dir / 'dataset.yaml'
    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        names = cfg.get('names', {})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        elif isinstance(names, dict):
            names = {int(k): v for k, v in names.items()}
        print(f"Loaded class names from {yaml_path}: {names}")
        return names
    print(f"No dataset.yaml found, using defaults: {DEFAULT_CLASS_NAMES}")
    return DEFAULT_CLASS_NAMES


def extract_crops(yolo_dir: str, output_dir: str, padding: int = 15,
                  min_crop_size: int = 8):
    """
    Extract crops from YOLO dataset into class-sorted train/val folders.
    
    Args:
        yolo_dir: Path to YOLO dataset root (with images/{train,val} and labels/{train,val})
        output_dir: Output directory for crops
        padding: Pixels to pad around each bounding box
        min_crop_size: Minimum crop dimension (skip tiny detections)
    """
    yolo_dir = Path(yolo_dir)
    output_dir = Path(output_dir)
    
    class_names = load_class_names(yolo_dir)
    
    stats = defaultdict(lambda: defaultdict(int))  # stats[split][class] = count
    total = 0
    skipped = 0
    
    for split in ['train', 'val', 'test']:
        img_dir = yolo_dir / 'images' / split
        lbl_dir = yolo_dir / 'labels' / split
        
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        
        # Gather all image files
        img_files = sorted(
            [f for f in img_dir.iterdir()
             if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
        )
        
        if not img_files:
            continue
        
        print(f"\n--- {split} split: {len(img_files)} images ---")
        
        # Create output class directories
        for cls_name in class_names.values():
            (output_dir / split / cls_name).mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(img_files, desc=f"  {split}"):
            # Find matching label file
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if not lbl_path.exists():
                continue
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Parse YOLO labels
            with open(lbl_path) as f:
                lines = f.read().strip().splitlines()
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                cls_id = int(float(parts[0]))
                cx_n, cy_n, bw_n, bh_n = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                # Convert normalized xywh → pixel xyxy
                cx, cy = cx_n * w, cy_n * h
                bw, bh = bw_n * w, bh_n * h
                
                x1 = int(cx - bw / 2 - padding)
                y1 = int(cy - bh / 2 - padding)
                x2 = int(cx + bw / 2 + padding)
                y2 = int(cy + bh / 2 + padding)
                
                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                if crop_w < min_crop_size or crop_h < min_crop_size:
                    skipped += 1
                    continue
                
                crop = img[y1:y2, x1:x2]
                
                cls_name = class_names.get(cls_id, f'class_{cls_id}')
                crop_filename = f"{img_path.stem}_crop{i:04d}.png"
                crop_path = output_dir / split / cls_name / crop_filename
                
                cv2.imwrite(str(crop_path), crop)
                stats[split][cls_name] += 1
                total += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("CROP EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Total crops: {total}   (skipped {skipped} tiny detections)")
    
    for split in ['train', 'val', 'test']:
        if split in stats:
            split_total = sum(stats[split].values())
            print(f"\n  {split}: {split_total} crops")
            for cls_name in sorted(stats[split]):
                print(f"    {cls_name:>10}: {stats[split][cls_name]}")
    
    print(f"{'='*60}\n")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Extract crops from YOLO dataset for EfficientNet training')
    parser.add_argument('--yolo-dir', type=str, required=True,
                        help='Path to YOLO dataset (with images/{train,val} and labels/{train,val})')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for crops')
    parser.add_argument('--padding', type=int, default=15,
                        help='Pixels of padding around each crop (default: 15)')
    parser.add_argument('--min-crop-size', type=int, default=8,
                        help='Minimum crop dimension in pixels (default: 8)')
    
    args = parser.parse_args()
    extract_crops(
        yolo_dir=args.yolo_dir,
        output_dir=args.output,
        padding=args.padding,
        min_crop_size=args.min_crop_size,
    )


if __name__ == '__main__':
    main()
