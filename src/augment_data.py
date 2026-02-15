"""
Enhanced Offline Data Augmentation for YOLO Training.

This script creates additional training images through augmentation when you have
a very small dataset. It generates augmented copies of images with corresponding
updated YOLO labels. Includes original data + augmented versions.

Features:
- Class-balanced oversampling (more augmentation for underrepresented classes)
- Smart augmentation that preserves small objects (fibers)
- Target-count mode for exact dataset sizes
- Progress tracking
- Original data included in output

Usage:
    # Target-count mode: exactly 800 train + 200 val images (recommended)
    python src/augment_data.py --input data/yolo --output data/yolo_augmented --target-train 800 --target-val 200
    
    # Standard augmentation (10x, medium severity)
    python src/augment_data.py --input data/yolo --output data/yolo_augmented --factor 10
    
    # Heavy augmentation for very small datasets (20x)
    python src/augment_data.py --input data/yolo --output data/yolo_augmented_heavy --factor 20 --severity heavy
    
    # Also augment validation set (useful for very small datasets)
    python src/augment_data.py --input data/yolo --output data/yolo_augmented --factor 10 --augment-val --val-factor 3
"""

import argparse
import cv2
import math
import numpy as np
from pathlib import Path
import shutil
import random
from typing import List, Tuple, Dict, Optional
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
                    cls = int(float(parts[0]))
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


def get_class_distribution(label_dir: Path) -> Dict[int, int]:
    """Count instances per class in a label directory."""
    class_counts = defaultdict(int)
    for label_file in label_dir.glob('*.txt'):
        labels = parse_yolo_label(str(label_file))
        for cls, _, _, _, _ in labels:
            class_counts[cls] += 1
    return dict(class_counts)


def get_image_class_info(input_path: Path, split: str) -> Dict[str, set]:
    """Get which classes each image contains."""
    image_classes = {}
    label_dir = input_path / 'labels' / split
    for label_file in label_dir.glob('*.txt'):
        labels = parse_yolo_label(str(label_file))
        classes_in_image = set(cls for cls, _, _, _, _ in labels)
        image_classes[label_file.stem] = classes_in_image
    return image_classes


def get_augmentation_pipeline(severity: str = 'medium', for_fibers: bool = False) -> A.Compose:
    """
    Create augmentation pipeline with bbox support.
    
    Args:
        severity: 'light', 'medium', or 'heavy'
        for_fibers: If True, use gentler transforms that preserve thin elongated objects
    """
    # Higher min_visibility for fibers (thin objects easily clipped)
    min_vis = 0.5 if for_fibers else 0.3
    
    if severity == 'light':
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=min_vis))
    
    elif severity == 'medium':
        if for_fibers:
            # Gentler pipeline for fiber-containing images
            # Avoids aggressive scaling/shearing that can destroy thin objects
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT, p=0.7),  # Full rotation is safe for fibers
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=35, val_shift_limit=25, p=0.6),
                A.OneOf([
                    A.CLAHE(clip_limit=3.0),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),  # Sharpening helps fiber edges
                ], p=0.5),
                A.GaussNoise(std_range=(0.03, 0.12), p=0.3),
                # Very gentle affine - no aggressive scaling/shearing
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(-0.05, 0.05),
                    shear=(-2, 2),
                    p=0.4
                ),
                A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=min_vis))
        else:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.GaussNoise(std_range=(0.03, 0.15), p=0.3),
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
            A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.7),
            A.OneOf([
                A.GaussNoise(std_range=(0.03, 0.2)),
                A.ISONoise(),
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
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=min_vis))
    
    return transform


def compute_class_weights(input_path: Path, split: str) -> Dict[str, float]:
    """
    Compute per-image augmentation weight based on class balance.
    
    Images containing underrepresented classes get higher weights,
    meaning they'll be augmented more times.
    """
    class_names = {0: 'fiber', 1: 'film', 2: 'fragment'}
    
    # Count total instances per class  
    class_counts = get_class_distribution(input_path / 'labels' / split)
    if not class_counts:
        return {}
    
    # Find the maximum class count (target for balancing)
    max_count = max(class_counts.values())
    
    # Compute class multipliers (how much to oversample each class)
    class_multiplier = {}
    for cls in class_counts:
        class_multiplier[cls] = max_count / class_counts[cls] if class_counts[cls] > 0 else 1.0
    
    print(f"\n  Class balancing weights:")
    for cls in sorted(class_counts.keys()):
        print(f"    {class_names.get(cls, f'class_{cls}')}: {class_counts[cls]} instances → {class_multiplier[cls]:.2f}x weight")
    
    # Get per-image class info
    image_classes = get_image_class_info(input_path, split)
    
    # Compute per-image weight = max multiplier of classes it contains
    image_weights = {}
    for stem, classes in image_classes.items():
        if classes:
            weight = max(class_multiplier.get(c, 1.0) for c in classes)
        else:
            weight = 1.0
        image_weights[stem] = weight
    
    return image_weights


def _augment_single_image(img_file: Path, input_path: Path, output_path: Path,
                          split: str, num_augments: int, transform: A.Compose,
                          fiber_transform: A.Compose, image_classes: Dict[str, set],
                          max_retries: int = 3) -> Tuple[int, int]:
    """
    Augment a single image. Returns (generated_count, skipped_count).
    Uses fiber-specific transform for images containing fibers.
    """
    generated = 0
    skipped = 0
    
    img = cv2.imread(str(img_file))
    if img is None:
        return 0, 0
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    label_file = input_path / 'labels' / split / f"{img_file.stem}.txt"
    labels = parse_yolo_label(str(label_file))
    bboxes, class_labels = yolo_to_albumentations(labels, height, width)
    
    # Copy original
    shutil.copy(img_file, output_path / 'images' / split / img_file.name)
    if label_file.exists():
        shutil.copy(label_file, output_path / 'labels' / split / label_file.name)
    
    # Choose transform based on whether image contains fibers
    has_fibers = 0 in image_classes.get(img_file.stem, set())
    active_transform = fiber_transform if has_fibers else transform
    
    # Generate augmented versions
    for i in range(num_augments):
        success = False
        for retry in range(max_retries):
            try:
                augmented = active_transform(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_labels = augmented['class_labels']
                
                # For fiber images: ensure fibers survived the augmentation
                if has_fibers and len(bboxes) > 0:
                    orig_fiber_count = sum(1 for c in class_labels if c == 0)
                    aug_fiber_count = sum(1 for c in aug_class_labels if c == 0)
                    # Retry if we lost more than 40% of fibers
                    if orig_fiber_count > 0 and aug_fiber_count < orig_fiber_count * 0.6:
                        if retry < max_retries - 1:
                            continue  # retry
                
                # General quality check: don't lose too many boxes
                if len(aug_bboxes) < len(bboxes) * 0.4 and len(bboxes) > 0:
                    if retry < max_retries - 1:
                        continue
                    skipped += 1
                    break
                
                # Convert back to YOLO format
                aug_labels = albumentations_to_yolo(aug_bboxes, aug_class_labels, height, width)
                
                # Save augmented image as JPEG (quality 95 - visually lossless, much smaller)
                aug_name = f"{img_file.stem}_aug{i:03d}.jpg"
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path / 'images' / split / aug_name), aug_img_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Save augmented labels
                aug_label_name = f"{img_file.stem}_aug{i:03d}.txt"
                save_yolo_label(aug_labels, str(output_path / 'labels' / split / aug_label_name))
                
                generated += 1
                success = True
                break
                
            except Exception as e:
                if retry == max_retries - 1:
                    skipped += 1
                continue
        
        if not success and i >= num_augments:
            break
    
    return generated, skipped


def augment_dataset(input_dir: str, output_dir: str, factor: int = 5, severity: str = 'medium',
                    augment_val: bool = False, val_factor: int = 3,
                    target_train: Optional[int] = None, target_val: Optional[int] = None):
    """
    Augment the YOLO dataset with class-balanced augmentation.
    
    Args:
        input_dir: Path to original YOLO dataset
        output_dir: Path to save augmented dataset (includes originals + augmented)
        factor: Number of augmented copies per original training image
        severity: Augmentation severity (light, medium, heavy)
        augment_val: Whether to also augment validation set
        val_factor: Number of augmented copies for validation images
        target_train: If set, target total number of training images (originals + augmented)
        target_val: If set, target total number of validation images (originals + augmented)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get class distribution
    print("\n" + "="*70)
    print("ENHANCED YOLO DATASET AUGMENTATION (Class-Balanced)")
    print("="*70)
    class_names = {0: 'fiber', 1: 'film', 2: 'fragment'}
    
    # --- TRAINING SET ---
    train_images = sorted((input_path / 'images' / 'train').glob('*.*'))
    train_images = [f for f in train_images if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}]
    
    train_class_dist = get_class_distribution(input_path / 'labels' / 'train')
    print("\nOriginal training class distribution:")
    for cls in sorted(train_class_dist.keys()):
        print(f"  Class {cls} ({class_names.get(cls, 'unknown')}): {train_class_dist[cls]} instances")
    print(f"  Total training images: {len(train_images)}")
    
    # Compute class-balanced weights for training images
    image_weights = compute_class_weights(input_path, 'train')
    image_classes = get_image_class_info(input_path, 'train')
    
    # Build augmentation pipelines
    transform = get_augmentation_pipeline(severity, for_fibers=False)
    fiber_transform = get_augmentation_pipeline(severity, for_fibers=True)
    
    # Auto-calculate factor from target if specified
    if target_train is not None:
        if target_train <= len(train_images):
            print(f"\n⚠️  Target train ({target_train}) <= original count ({len(train_images)}). Will copy originals only.")
            factor = 0
        else:
            # Base factor, will be adjusted per-image by class weights
            augments_needed = target_train - len(train_images)
            # Compute weighted total to determine base factor
            total_weight = sum(image_weights.get(f.stem, 1.0) for f in train_images)
            # base_factor * total_weight ≈ augments_needed
            base_factor = augments_needed / total_weight if total_weight > 0 else factor
            # Over-generate slightly for trimming
            base_factor = base_factor * 1.15
            factor = max(1, math.ceil(base_factor))
    
    if target_val is not None:
        augment_val = True
        val_images = sorted((input_path / 'images' / 'val').glob('*.*'))
        val_images = [f for f in val_images if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}]
        val_images_count = len(val_images)
        if target_val <= val_images_count:
            val_factor = 0
        else:
            val_factor = math.ceil((target_val - val_images_count) / val_images_count) + 1
    
    print(f"\nTraining augmentation plan:")
    print(f"  Base factor: {factor}x (adjusted per-image by class weight)")
    print(f"  Severity: {severity}")
    if target_train:
        print(f"  Target total: {target_train} images")
    
    # Process training images with class-balanced augmentation
    total_generated = 0
    total_skipped = 0
    
    print(f"\nProcessing training images (class-balanced)...")
    for idx, img_file in enumerate(train_images, 1):
        # Per-image augmentation count based on class weight
        weight = image_weights.get(img_file.stem, 1.0)
        num_augments = max(1, round(factor * weight))
        
        generated, skipped = _augment_single_image(
            img_file, input_path, output_path, 'train',
            num_augments, transform, fiber_transform, image_classes
        )
        total_generated += generated
        total_skipped += skipped
        
        classes_str = ','.join(class_names.get(c, '?') for c in sorted(image_classes.get(img_file.stem, set())))
        print(f"  [{idx}/{len(train_images)}] {img_file.name} [{classes_str}] → {generated} augmented (weight={weight:.1f}x)")
    
    # --- VALIDATION SET ---
    val_images = sorted((input_path / 'images' / 'val').glob('*.*'))
    val_images = [f for f in val_images if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}]
    
    if augment_val and val_images:
        val_image_classes = get_image_class_info(input_path, 'val')
        print(f"\nProcessing validation images (augmentation enabled)...")
        print(f"  Original images: {len(val_images)}")
        print(f"  Augmentation factor: {val_factor}x")
        
        val_generated = 0
        for idx, img_file in enumerate(val_images, 1):
            generated, _ = _augment_single_image(
                img_file, input_path, output_path, 'val',
                val_factor, transform, fiber_transform, val_image_classes
            )
            val_generated += generated
            print(f"  [{idx}/{len(val_images)}] {img_file.name} → {generated} augmented")
        
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
    
    # Trim to exact target counts if specified
    if target_train is not None:
        _trim_to_target(output_path / 'images' / 'train', output_path / 'labels' / 'train',
                        target_train, train_images, 'train')
    
    if target_val is not None:
        orig_val_names = [f.name for f in (input_path / 'images' / 'val').glob('*.*')]
        _trim_to_target(output_path / 'images' / 'val', output_path / 'labels' / 'val',
                        target_val, None, 'val', orig_val_names)
    
    # Count final images and class distribution
    final_train = len(list((output_path / 'images' / 'train').glob('*.*')))
    final_val = len(list((output_path / 'images' / 'val').glob('*.*')))
    final_train_dist = get_class_distribution(output_path / 'labels' / 'train')
    final_val_dist = get_class_distribution(output_path / 'labels' / 'val')
    
    print(f"\n{'='*70}")
    print("✅ AUGMENTATION COMPLETE")
    print(f"{'='*70}")
    
    print(f"\nTraining set:")
    print(f"  Original images:     {len(train_images)}")
    print(f"  Augmented generated: {total_generated}")
    print(f"  Skipped (quality):   {total_skipped}")
    print(f"  Total in output:     {final_train}")
    if target_train:
        ok = '✅ YES' if final_train == target_train else f'⚠️ NO ({final_train}/{target_train})'
        print(f"  Target achieved:     {ok}")
    
    print(f"\n  Final class distribution (train):")
    for cls in sorted(final_train_dist.keys()):
        orig = train_class_dist.get(cls, 0)
        final = final_train_dist[cls]
        print(f"    {class_names.get(cls, f'class_{cls}')}: {orig} → {final} instances ({final/orig:.1f}x)" if orig > 0 else f"    {class_names.get(cls, f'class_{cls}')}: {final} instances")
    
    print(f"\nValidation set:")
    print(f"  Total in output:     {final_val}")
    if target_val:
        ok = '✅ YES' if final_val == target_val else f'⚠️ NO ({final_val}/{target_val})'
        print(f"  Target achieved:     {ok}")
    
    if final_val_dist:
        print(f"\n  Final class distribution (val):")
        for cls in sorted(final_val_dist.keys()):
            print(f"    {class_names.get(cls, f'class_{cls}')}: {final_val_dist[cls]} instances")
    
    print(f"\nOutput:")
    print(f"  Directory: {output_path}")
    print(f"  Config:    {output_path / 'dataset.yaml'}")
    
    print(f"\n💡 Next steps:")
    print(f"  Train with augmented data:")
    print(f"    python src/train_yolo.py --mode train --data {output_path}/dataset.yaml \\")
    print(f"      --epochs 300 --model-size m --batch 8 --imgsz 1280")
    print(f"{'='*70}\n")


def _trim_to_target(images_dir: Path, labels_dir: Path, target: int,
                    original_images=None, split_name: str = '', orig_names: list = None):
    """
    Trim augmented images to hit exact target count.
    Only removes augmented images (with '_aug' in name), never originals.
    """
    all_images = sorted(images_dir.glob('*.*'))
    current_count = len(all_images)
    
    if current_count <= target:
        if current_count < target:
            print(f"  ⚠️  {split_name}: Only generated {current_count}/{target} images (not enough source images)")
        return
    
    # Build set of original filenames to protect
    if orig_names is not None:
        protected = set(orig_names)
    elif original_images is not None:
        protected = {f.name for f in original_images}
    else:
        protected = set()
    
    # Separate augmented from originals
    augmented = [f for f in all_images if f.name not in protected]
    
    excess = current_count - target
    if excess > len(augmented):
        print(f"  ⚠️  {split_name}: Cannot trim enough (need to remove {excess} but only {len(augmented)} augmented)")
        excess = len(augmented)
    
    # Randomly select which augmented images to remove
    to_remove = random.sample(augmented, excess)
    for img_file in to_remove:
        img_file.unlink()
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            label_file.unlink()
    
    print(f"  Trimmed {split_name}: removed {excess} surplus augmented images → {target} total")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced YOLO Dataset Augmentation for Microplastic Detection (Class-Balanced)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Target-count mode: exactly 800 train + 200 val images (recommended)
  python src/augment_data.py --input data/yolo --output data/yolo_augmented --target-train 800 --target-val 200
  
  # Standard factor-based augmentation (10x, medium severity)
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
    parser.add_argument('--target-train', type=int, default=None,
                        help='Target total training images (originals + augmented). Overrides --factor.')
    parser.add_argument('--target-val', type=int, default=None,
                        help='Target total validation images (originals + augmented). Enables val augmentation.')
    
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
    
    augment_dataset(args.input, args.output, args.factor, args.severity, 
                    args.augment_val, args.val_factor,
                    args.target_train, args.target_val)


if __name__ == "__main__":
    main()
