"""
Augment YOLO dataset for Microplastics in Soil.
Target 800 train images and 200 validation images.

Usage:
    python src/data_preparation/augment_data_micro.py --input data/micro/yolo_single --output data/micro/yolo_single_aug --target-train 800 --target-val 200
"""

import argparse
import shutil
import cv2
import yaml
import math
import random
from pathlib import Path
import albumentations as A
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Augment Microplastic YOLO dataset to target sizes')
    parser.add_argument('--input', type=str, required=True, help='Input dataset directory')
    parser.add_argument('--output', type=str, required=True, help='Output augmented directory')
    parser.add_argument('--target-train', type=int, default=800, help='Target number of train images')
    parser.add_argument('--target-val', type=int, default=200, help='Target number of val images')
    return parser.parse_args()

def get_micro_augmentation_pipeline():
    """
    Augmentation pipeline optimized for microplastics on soil surfaces.
    """
    return A.Compose([
        # Spatial transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=45, p=0.8),
        
        # Color and lighting (soil lighting variability)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),
        
        # Noise and blur (microscope/camera irregularities)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=1.0),
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0)
        ], p=0.4),
        
        # Occlusion (simulating being partially buried in soil)
        A.CoarseDropout(max_holes=6, max_height=8, max_width=8, min_holes=1, min_height=4, min_width=4, fill_value=0, p=0.2),
        
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=16, min_visibility=0.3))

def load_yolo_labels(label_path):
    labels = []
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    labels.append((c, x, y, w, h))
    return labels

def save_yolo_labels(labels, output_path):
    with open(output_path, 'w') as f:
        for c, x, y, w, h in labels:
            f.write(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def process_split(input_dir, output_dir, split, target_count, transform):
    img_in_dir = input_dir / 'images' / split
    lbl_in_dir = input_dir / 'labels' / split
    img_out_dir = output_dir / 'images' / split
    lbl_out_dir = output_dir / 'labels' / split

    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    if not img_in_dir.exists():
        print(f"Skipping {split}, images dir missing.")
        return

    images = list(img_in_dir.glob('*.*'))
    images = [f for f in images if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}]
    
    orig_count = len(images)
    print(f"[{split.upper()}] Found {orig_count} original images. Target: {target_count}")

    if orig_count == 0:
        return
        
    if orig_count >= target_count:
        print(f"[{split.upper()}] We already have enough. Just copying first {target_count}.")
        selected_images = images[:target_count]
        for img_path in tqdm(selected_images):
            shutil.copy(img_path, img_out_dir / img_path.name)
            lbl_path = lbl_in_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_out_dir / lbl_path.name)
            else:
                (lbl_out_dir / lbl_path.name).write_text("")
        return

    # 1. Copy all originals first
    print(f"[{split.upper()}] Copying originals...")
    for img_path in images:
        shutil.copy(img_path, img_out_dir / img_path.name)
        lbl_path = lbl_in_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy(lbl_path, lbl_out_dir / lbl_path.name)
        else:
            (lbl_out_dir / lbl_path.name).write_text("")

    # 2. Generate augmentations to reach target count
    needed = target_count - orig_count
    print(f"[{split.upper()}] Need {needed} specific augmentations.")
    
    # Calculate how many augs per image roughly
    augs_per_image = math.ceil(needed / orig_count)
    
    generated = 0
    pbar = tqdm(total=needed)
    
    # Loop over originals and augment until we hit the exact target
    idx = 0
    while generated < needed:
        img_path = images[idx % orig_count]
        idx += 1
        
        lbl_path = lbl_in_dir / f"{img_path.stem}.txt"
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        yolo_labels = load_yolo_labels(lbl_path)
        
        bboxes = [[l[1], l[2], l[3], l[4]] for l in yolo_labels]
        class_labels = [l[0] for l in yolo_labels]
        
        aug_name = f"{img_path.stem}_aug_{generated}{img_path.suffix}"
        
        # Try augmentation
        try:
            augmented = transform(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
            
            aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            aug_bboxes = augmented['bboxes']
            aug_classes = augmented['class_labels']
            
            # Save augmented image
            cv2.imwrite(str(img_out_dir / aug_name), aug_img)
            
            # Save augmented labels
            out_lbl = []
            for bbox, c in zip(aug_bboxes, aug_classes):
                out_lbl.append((c, bbox[0], bbox[1], bbox[2], bbox[3]))
            save_yolo_labels(out_lbl, lbl_out_dir / f"{Path(aug_name).stem}.txt")
            
            generated += 1
            pbar.update(1)
        except Exception as e:
            # Albumentations could fail on bad bounding boxes clipping, just skip
            continue
            
    pbar.close()
    print(f"[{split.upper()}] Done. Reached {orig_count + generated} images.")

def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return
        
    print(f"\n{'='*60}")
    print("AUGMENT MICROPLASTIC YOLO DATASET")
    print(f"{'='*60}")
    
    transform = get_micro_augmentation_pipeline()
    
    process_split(input_dir, output_dir, 'train', args.target_train, transform)
    process_split(input_dir, output_dir, 'val', args.target_val, transform)
    
    # Copy dataset.yaml and modify path
    yaml_in = input_dir / 'dataset.yaml'
    if yaml_in.exists():
        with open(yaml_in, 'r') as f:
            config = yaml.safe_load(f)
        config['path'] = str(output_dir.resolve())
        with open(output_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
    # Copy classes.txt
    classes_in = input_dir / 'classes.txt'
    if classes_in.exists():
        shutil.copy(classes_in, output_dir / 'classes.txt')
        
    print(f"\nDone! Augmented dataset saved to {output_dir}")

if __name__ == '__main__':
    main()
