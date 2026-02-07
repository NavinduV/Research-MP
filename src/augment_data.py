"""
Offline Data Augmentation for YOLO Training.

This script creates additional training images through augmentation when you have
a very small dataset. It generates augmented copies of images with corresponding
updated YOLO labels.

Usage:
    python src/augment_data.py --input data/yolo --output data/yolo_augmented --factor 10
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import shutil
import random
from typing import List, Tuple
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


def get_augmentation_pipeline(severity: str = 'medium') -> A.Compose:
    """Create augmentation pipeline with bbox support."""
    
    if severity == 'light':
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
    
    elif severity == 'medium':
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.5),
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
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
    
    else:  # heavy
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT, p=0.7),
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
                scale=(0.8, 1.2),
                translate_percent=(-0.15, 0.15),
                shear=(-10, 10),
                p=0.6
            ),
            A.Perspective(scale=(0.02, 0.08), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.2))
    
    return transform


def augment_dataset(input_dir: str, output_dir: str, factor: int = 5, severity: str = 'medium'):
    """
    Augment the YOLO dataset.
    
    Args:
        input_dir: Path to original YOLO dataset
        output_dir: Path to save augmented dataset
        factor: Number of augmented copies per original image
        severity: Augmentation severity (light, medium, heavy)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    transform = get_augmentation_pipeline(severity)
    
    # Process training data (augment only training set)
    train_images = list((input_path / 'images' / 'train').glob('*.*'))
    print(f"\nFound {len(train_images)} training images")
    print(f"Generating {factor} augmented copies per image...")
    print(f"Augmentation severity: {severity}")
    print(f"Total output images: {len(train_images) * (factor + 1)}\n")
    
    total_generated = 0
    
    for img_file in train_images:
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not load {img_file}")
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
        for i in range(factor):
            try:
                # Apply augmentation
                augmented = transform(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_labels = augmented['class_labels']
                
                # Skip if too many boxes were lost
                if len(aug_bboxes) < len(bboxes) * 0.5 and len(bboxes) > 0:
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
                
            except Exception as e:
                print(f"Warning: Augmentation failed for {img_file.name} (aug {i}): {e}")
                continue
        
        print(f"Processed: {img_file.name}")
    
    # Copy validation and test sets unchanged
    for split in ['val', 'test']:
        for img_file in (input_path / 'images' / split).glob('*.*'):
            shutil.copy(img_file, output_path / 'images' / split / img_file.name)
        for label_file in (input_path / 'labels' / split).glob('*.txt'):
            shutil.copy(label_file, output_path / 'labels' / split / label_file.name)
    
    # Create dataset.yaml
    dataset_config = f"""
path: {output_path.absolute()}
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
        f.write(dataset_config.strip())
    
    print(f"\n{'='*60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Original training images: {len(train_images)}")
    print(f"New augmented images: {total_generated}")
    print(f"Total training images: {len(train_images) + total_generated}")
    print(f"Output directory: {output_path}")
    print(f"Dataset config: {output_path / 'dataset.yaml'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Augment YOLO dataset for microplastic detection')
    parser.add_argument('--input', type=str, default='data/yolo', 
                        help='Path to input YOLO dataset')
    parser.add_argument('--output', type=str, default='data/yolo_augmented',
                        help='Path to output augmented dataset')
    parser.add_argument('--factor', type=int, default=10,
                        help='Number of augmented copies per image')
    parser.add_argument('--severity', type=str, choices=['light', 'medium', 'heavy'],
                        default='medium', help='Augmentation severity')
    
    args = parser.parse_args()
    
    # Check albumentations is installed
    try:
        import albumentations
    except ImportError:
        print("Error: albumentations not installed. Run: pip install albumentations")
        return
    
    augment_dataset(args.input, args.output, args.factor, args.severity)


if __name__ == "__main__":
    main()
