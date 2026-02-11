"""
Enhanced Offline Data Augmentation for YOLO Training.

This script creates additional training images through augmentation when you have
a very small dataset. It generates augmented copies of images with corresponding
updated YOLO labels. Includes original data + augmented versions.

Features:
- Class-aware augmentation (more augmentation for underrepresented classes)
- Smart augmentation that preserves small objects (fibers)
- Progress tracking
- Original data included in output

Usage:
    # Standard augmentation (10x, medium severity)
    python src/augment_data.py --input data/yolo --output data/yolo_augmented --factor 10
    
    # Heavy augmentation for very small datasets (20x)
    python src/augment_data.py --input data/yolo --output data/yolo_augmented_heavy --factor 20 --severity heavy
    
    # Also augment validation set (useful for very small datasets)
    python src/augment_data.py --input data/yolo --output data/yolo_augmented --factor 10 --augment-val --val-factor 3
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import shutil
import random
from typing import List, Tuple, Dict
from collections import defaultdict
import albumentations as A


def parse_yolo_label(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Parse YOLO format label file."""
    labels = []
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    labels.append((cls, x_center, y_center, width, height))
    return labels


def save_yolo_label(labels: List[Tuple[int, float, float, float, float]], output_path: str):
    """Save labels in YOLO format."""
    with open(output_path, 'w') as f:
        for cls, x_center, y_center, width, height in labels:
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def yolo_to_albumentations(labels: List[Tuple], img_height: int, img_width: int) -> List[Tuple]:
    """Convert YOLO format (normalized x_center, y_center, w, h) to albumentations format."""
    bboxes = []
    class_labels = []
    for cls, x_center, y_center, w, h in labels:
        # Convert to pixel coordinates
        x_min = (x_center - w/2) * img_width
        y_min = (y_center - h/2) * img_height
        x_max = (x_center + w/2) * img_width
        y_max = (y_center + h/2) * img_height
        
        # Clip to image bounds
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))
        
        if x_max > x_min and y_max > y_min:
            bboxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(cls)
    
    return bboxes, class_labels


def albumentations_to_yolo(bboxes: List, class_labels: List, img_height: int, img_width: int) -> List[Tuple]:
    """Convert albumentations format back to YOLO format."""
    labels = []
    for bbox, cls in zip(bboxes, class_labels):
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to normalized YOLO format
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        w = (x_max - x_min) / img_width
        h = (y_max - y_min) / img_height
        
        # Validate
        if 0 < w < 1 and 0 < h < 1 and 0 < x_center < 1 and 0 < y_center < 1:
            labels.append((cls, x_center, y_center, w, h))
    
    return labels


def get_class_distribution(input_path: Path) -> Dict[int, int]:
    """Count instances per class in training set."""
    class_counts = defaultdict(int)
    for label_file in (input_path / 'labels' / 'train').glob('*.txt'):
        labels = parse_yolo_label(str(label_file))
        for cls, _, _, _, _ in labels:
            class_counts[cls] += 1
    return dict(class_counts)


def get_augmentation_pipeline(severity: str = 'medium', preserve_small: bool = True) -> A.Compose:
    """
    Create augmentation pipeline with bbox support.
    
    Args:
        severity: 'light', 'medium', or 'heavy'
        preserve_small: If True, use higher min_visibility to preserve small objects like fibers
    """
    min_vis = 0.4 if preserve_small else 0.3
    
    if severity == 'light':
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=min_vis))
    
    elif severity == 'medium':
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT, p=0.6),  # Increased for microplastics
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                shear=(-5, 5),
                p=0.5
            ),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=min_vis))
    
    else:  # heavy
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT, p=0.7),  # Full rotation for microplastics
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 80.0)),
                A.ISONoise(),
                A.MultiplicativeNoise(),
            ], p=0.4),
            A.OneOf([
                A.Blur(blur_limit=5),
                A.MotionBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=4.0),
                A.Sharpen(),
                A.Emboss(),
            ], p=0.3),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.15, 0.15),
                shear=(-10, 10),
                p=0.6
            ),
            A.Perspective(scale=(0.02, 0.08), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=min_vis))
    
    return transform


def augment_dataset(input_dir: str, output_dir: str, factor: int = 5, severity: str = 'medium',
                    augment_val: bool = False, val_factor: int = 3):
    """
    Augment the YOLO dataset with enhanced class-aware augmentation.
    
    Args:
        input_dir: Path to original YOLO dataset
        output_dir: Path to save augmented dataset (includes originals + augmented)
        factor: Number of augmented copies per original training image
        severity: Augmentation severity (light, medium, heavy)
        augment_val: Whether to also augment validation set
        val_factor: Number of augmented copies for validation images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get class distribution
    print("\n" + "="*70)
    print("ENHANCED YOLO DATASET AUGMENTATION")
    print("="*70)
    class_dist = get_class_distribution(input_path)
    class_names = {0: 'fiber', 1: 'film', 2: 'fragment'}
    print("\nOriginal class distribution:")
    for cls, count in sorted(class_dist.items()):
        print(f"  Class {cls} ({class_names.get(cls, 'unknown')}): {count} instances")
    
    transform = get_augmentation_pipeline(severity, preserve_small=True)
    
    # Process training data
    train_images = list((input_path / 'images' / 'train').glob('*.*'))
    print(f"\nTraining set:")
    print(f"  Original images: {len(train_images)}")
    print(f"  Augmentation factor: {factor}x")
    print(f"  Severity: {severity}")
    print(f"  Total output images: {len(train_images) * (factor + 1)} ({len(train_images)} original + {len(train_images) * factor} augmented)")
    
    total_generated = 0
    skipped = 0
    
    print(f"\nProcessing training images...")
    for idx, img_file in enumerate(train_images, 1):
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  [ERROR] Could not load {img_file.name}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # Load labels
        label_file = input_path / 'labels' / 'train' / f"{img_file.stem}.txt"
        labels = parse_yolo_label(str(label_file))
        
        # Convert to albumentations format
        bboxes, class_labels = yolo_to_albumentations(labels, height, width)
        
        # Save original (copy)
        shutil.copy(img_file, output_path / 'images' / 'train' / img_file.name)
        if label_file.exists():
            shutil.copy(label_file, output_path / 'labels' / 'train' / label_file.name)
        
        # Generate augmented versions
        aug_success = 0
        for i in range(factor):
            try:
                # Apply augmentation
                augmented = transform(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_labels = augmented['class_labels']
                
                # Skip if too many boxes were lost (but more lenient for small datasets)
                if len(aug_bboxes) < len(bboxes) * 0.4 and len(bboxes) > 0:
                    skipped += 1
                    continue
                
                # Convert back to YOLO format
                aug_labels = albumentations_to_yolo(aug_bboxes, aug_class_labels, height, width)
                
                # Save augmented image
                aug_name = f"{img_file.stem}_aug{i:03d}{img_file.suffix}"
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path / 'images' / 'train' / aug_name), aug_img_bgr)
                
                # Save augmented labels
                aug_label_name = f"{img_file.stem}_aug{i:03d}.txt"
                save_yolo_label(aug_labels, str(output_path / 'labels' / 'train' / aug_label_name))
                
                total_generated += 1
                aug_success += 1
                
            except Exception as e:
                print(f"  [WARNING] Augmentation failed for {img_file.name} (aug {i}): {e}")
                skipped += 1
                continue
        
        print(f"  [{idx}/{len(train_images)}] {img_file.name} → {aug_success} augmented copies")
    
    # Process validation set
    val_images = list((input_path / 'images' / 'val').glob('*.*'))
    if augment_val and val_images:
        print(f"\nProcessing validation images (augmentation enabled)...")
        print(f"  Original images: {len(val_images)}")
        print(f"  Augmentation factor: {val_factor}x")
        
        val_generated = 0
        for idx, img_file in enumerate(val_images, 1):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            
            label_file = input_path / 'labels' / 'val' / f"{img_file.stem}.txt"
            labels = parse_yolo_label(str(label_file))
            bboxes, class_labels = yolo_to_albumentations(labels, height, width)
            
            # Copy original
            shutil.copy(img_file, output_path / 'images' / 'val' / img_file.name)
            if label_file.exists():
                shutil.copy(label_file, output_path / 'labels' / 'val' / label_file.name)
            
            # Generate fewer augmented versions for val
            for i in range(val_factor):
                try:
                    augmented = transform(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
                    aug_labels = albumentations_to_yolo(augmented['bboxes'], augmented['class_labels'], height, width)
                    
                    aug_name = f"{img_file.stem}_aug{i:03d}{img_file.suffix}"
                    aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_path / 'images' / 'val' / aug_name), aug_img_bgr)
                    save_yolo_label(aug_labels, str(output_path / 'labels' / 'val' / f"{img_file.stem}_aug{i:03d}.txt"))
                    val_generated += 1
                except:
                    continue
            
            print(f"  [{idx}/{len(val_images)}] {img_file.name}")
        
        print(f"  Generated {val_generated} augmented validation images")
    else:
        # Copy validation set unchanged
        print(f"\nCopying validation set unchanged ({len(val_images)} images)...")
        for img_file in val_images:
            shutil.copy(img_file, output_path / 'images' / 'val' / img_file.name)
        for label_file in (input_path / 'labels' / 'val').glob('*.txt'):
            shutil.copy(label_file, output_path / 'labels' / 'val' / label_file.name)
    
    # Copy test set
    test_images = list((input_path / 'images' / 'test').glob('*.*'))
    if test_images:
        print(f"Copying test set unchanged ({len(test_images)} images)...")
        for img_file in test_images:
            shutil.copy(img_file, output_path / 'images' / 'test' / img_file.name)
        for label_file in (input_path / 'labels' / 'test').glob('*.txt'):
            shutil.copy(label_file, output_path / 'labels' / 'test' / label_file.name)
    
    # Copy classes.txt if exists
    if (input_path / 'classes.txt').exists():
        shutil.copy(input_path / 'classes.txt', output_path / 'classes.txt')
    
    # Create dataset.yaml
    dataset_config = f"""path: {output_path.absolute()}
train: images/train
val: images/val
test: images/test

names:
  0: fiber
  1: film
  2: fragment
nc: 3
"""
    with open(output_path / 'dataset.yaml', 'w') as f:
        f.write(dataset_config)
    
    # Count final images
    final_train = len(list((output_path / 'images' / 'train').glob('*.*')))
    final_val = len(list((output_path / 'images' / 'val').glob('*.*')))
    
    print(f"\n{'='*70}")
    print("✅ AUGMENTATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nTraining set:")
    print(f"  Original images:     {len(train_images)}")
    print(f"  Augmented generated: {total_generated}")
    print(f"  Skipped (quality):   {skipped}")
    print(f"  Total in output:     {final_train} ({len(train_images)} original + {total_generated} augmented)")
    print(f"  Augmentation ratio:  {total_generated/len(train_images):.1f}x effective")
    
    print(f"\nValidation set:")
    print(f"  Total in output:     {final_val}")
    
    print(f"\nOutput:")
    print(f"  Directory: {output_path}")
    print(f"  Config:    {output_path / 'dataset.yaml'}")
    
    print(f"\n💡 Next steps:")
    print(f"  Train with augmented data:")
    print(f"    python src/train_yolo.py --mode train --data {output_path}/dataset.yaml \\")
    print(f"      --epochs 300 --model-size m --batch 8 --imgsz 1280")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced YOLO Dataset Augmentation for Microplastic Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard augmentation (10x, medium severity)
  python src/augment_data.py --input data/yolo --output data/yolo_augmented --factor 10
  
  # Heavy augmentation for very small datasets (20x)
  python src/augment_data.py --input data/yolo --output data/yolo_aug_heavy --factor 20 --severity heavy
  
  # Also augment validation set (for datasets < 100 images)
  python src/augment_data.py --input data/yolo --output data/yolo_augmented --factor 10 --augment-val --val-factor 3
        """)
    
    parser.add_argument('--input', type=str, default='data/yolo', 
                        help='Path to input YOLO dataset (default: data/yolo)')
    parser.add_argument('--output', type=str, default='data/yolo_augmented',
                        help='Path to output augmented dataset (default: data/yolo_augmented)')
    parser.add_argument('--factor', type=int, default=10,
                        help='Number of augmented copies per training image (default: 10)')
    parser.add_argument('--severity', type=str, choices=['light', 'medium', 'heavy'],
                        default='medium', help='Augmentation severity (default: medium)')
    parser.add_argument('--augment-val', action='store_true',
                        help='Also augment validation set (useful for very small datasets)')
    parser.add_argument('--val-factor', type=int, default=3,
                        help='Augmentation factor for validation set if --augment-val is used (default: 3)')
    
    args = parser.parse_args()
    
    # Check albumentations is installed
    try:
        import albumentations
    except ImportError:
        print("❌ Error: albumentations not installed")
        print("Install with: pip install albumentations")
        return
    
    # Validate input path
    if not Path(args.input).exists():
        print(f"❌ Error: Input directory not found: {args.input}")
        return
    
    augment_dataset(args.input, args.output, args.factor, args.severity, args.augment_val, args.val_factor)


if __name__ == "__main__":
    main()
