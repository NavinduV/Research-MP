"""
Convert multi-class YOLO dataset to single-class ("microplastic").

Merges all class IDs (fiber=0, film=1, fragment=2) into a single class (0=microplastic).
Creates a new dataset directory with updated labels and a new dataset.yaml.

Usage:
    # Convert original dataset
    python src/data_preparation/convert_yolo_single_class.py --input data/yolo --output data/yolo_single

    # Convert augmented dataset
    python src/data_preparation/convert_yolo_single_class.py --input data/yolo_augmented --output data/yolo_augmented_single
"""

import argparse
import shutil
from pathlib import Path
import yaml


def convert_labels(input_dir: Path, output_dir: Path):
    """Convert all label files: replace class ID with 0."""
    converted = 0
    total_boxes = 0

    for split in ('train', 'val', 'test'):
        src_label_dir = input_dir / 'labels' / split
        dst_label_dir = output_dir / 'labels' / split
        if not src_label_dir.exists():
            continue
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        for label_file in sorted(src_label_dir.glob('*.txt')):
            lines = label_file.read_text().strip().split('\n')
            new_lines = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.strip().split()
                # Replace class ID with 0
                parts[0] = '0'
                new_lines.append(' '.join(parts))
                total_boxes += 1

            (dst_label_dir / label_file.name).write_text('\n'.join(new_lines) + '\n')
            converted += 1

    return converted, total_boxes


def copy_images(input_dir: Path, output_dir: Path):
    """Copy image directories (symlink-safe)."""
    for split in ('train', 'val', 'test'):
        src_img_dir = input_dir / 'images' / split
        dst_img_dir = output_dir / 'images' / split
        if not src_img_dir.exists():
            continue
        if dst_img_dir.exists():
            shutil.rmtree(dst_img_dir)
        shutil.copytree(src_img_dir, dst_img_dir)


def create_dataset_yaml(output_dir: Path):
    """Create single-class dataset.yaml."""
    config = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'microplastic'},
        'nc': 1,
    }
    with open(output_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO dataset to single-class')
    parser.add_argument('--input', type=str, required=True, help='Input YOLO dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output single-class dataset directory')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    print(f"\n{'='*60}")
    print("CONVERT YOLO DATASET TO SINGLE-CLASS")
    print(f"{'='*60}")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    print("[1/3] Copying images...")
    copy_images(input_dir, output_dir)

    # Convert labels
    print("[2/3] Converting labels (all classes -> 0:microplastic)...")
    num_files, num_boxes = convert_labels(input_dir, output_dir)
    print(f"  Converted {num_files} label files ({num_boxes} bounding boxes)")

    # Create dataset.yaml
    print("[3/3] Creating dataset.yaml...")
    create_dataset_yaml(output_dir)

    # Copy classes.txt
    (output_dir / 'classes.txt').write_text('microplastic\n')

    print(f"\nDone! Single-class dataset at: {output_dir}")
    print(f"  dataset.yaml: {output_dir / 'dataset.yaml'}")
    print(f"\nNext steps:")
    print(f"  1. Train YOLO:  python src/train/train_yolo.py --mode train --data {output_dir}/dataset.yaml --epochs 100 --model-size l --batch 8 --imgsz 1280")
    print(f"  2. Train EfficientNet (already trained or retrain on crops)")


if __name__ == '__main__':
    main()
