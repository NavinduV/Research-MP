"""
Convert multi-class YOLO dataset (flat structure) to single-class ("microplastic"),
and additionally perform an 80/20 train/val split.

Usage:
    python src/data_preparation/convert_yolo_single_class_micro.py --input data/micro/yolo --output data/micro/yolo_single
"""

import argparse
import shutil
import random
from pathlib import Path
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Convert flat YOLO dataset to single-class with train/val split')
    parser.add_argument('--input', type=str, required=True, help='Input YOLO dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output single-class dataset directory')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Ratio for training set')
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    print(f"\n{'='*60}")
    print("CONVERT & SPLIT YOLO DATASET TO SINGLE-CLASS (MICRO)")
    print(f"{'='*60}")
    print(f"  Input        : {input_dir}")
    print(f"  Output       : {output_dir}")
    print(f"  Train Ratio  : {args.split_ratio}")
    print(f"{'='*60}\n")

    # Clear/create output directories
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Gather images
    image_dir = input_dir / 'images'
    images = []
    if image_dir.exists():
        images = [f for f in image_dir.glob('*.*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}]
    
    if not images:
        print("ERROR: No images found in input directory.")
        return

    random.seed(42)
    random.shuffle(images)

    split_idx = int(len(images) * args.split_ratio)
    splits = {
        'train': images[:split_idx],
        'val': images[split_idx:]
    }

    print(f"[1/3] Splitting and copying images (Total: {len(images)})")
    for split, img_list in splits.items():
        print(f"  -> {split}: {len(img_list)} images")
        for img in img_list:
            shutil.copy(img, output_dir / 'images' / split / img.name)

    print("\n[2/3] Converting and splitting labels (all classes -> 0:microplastic)...")
    total_converted = 0
    total_boxes = 0

    label_dir = input_dir / 'labels'
    for split, img_list in splits.items():
        for img in img_list:
            label_file = label_dir / f"{img.stem}.txt"
            out_label = output_dir / 'labels' / split / label_file.name

            if label_file.exists():
                old_lines = label_file.read_text().strip().split('\n')
                new_lines = []
                for line in old_lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    parts[0] = '0'  # force class to 0
                    new_lines.append(' '.join(parts))
                    total_boxes += 1
                
                out_label.write_text('\n'.join(new_lines) + '\n')
                total_converted += 1
            else:
                # Create empty label file for negative samples
                out_label.write_text('')

    print(f"  Converted {total_converted} label files with {total_boxes} bounding boxes")

    print("\n[3/3] Creating dataset.yaml...")
    config = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'microplastic'},
        'nc': 1,
    }
    with open(output_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    (output_dir / 'classes.txt').write_text('microplastic\n')

    print(f"\nDone! Single-class dataset created at: {output_dir}")

if __name__ == '__main__':
    main()
