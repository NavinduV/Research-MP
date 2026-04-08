import argparse
import random
import shutil
import json
from pathlib import Path

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


def get_augmenter():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2)
    ])


def maybe_upscale(img, upscale_factor=1.0):
    """Upscale image by factor (>1.0). Returns original image when factor is 1.0."""
    if upscale_factor <= 1.0:
        return img

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * upscale_factor)))
    new_h = max(1, int(round(h * upscale_factor)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def augment_and_save(input_dir, output_dir, train_target=800, val_target=200, upscale_factor=1.0):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    classes = ['fiber', 'film', 'fragment']
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    for cls in classes:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
        
    augmenter = get_augmenter()
    annotations = {}
    
    for cls_id, cls in enumerate(classes):
        cls_dir = input_dir / cls
        if not cls_dir.exists():
            continue
            
        images = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        # Make a stable split
        images = sorted(images)
        random.seed(42)
        random.shuffle(images)
        
        # Original train/val split (80/20)
        val_size = int(len(images) * 0.2)
        val_images = images[:val_size]
        train_images = images[val_size:]
        
        # Save original images and record annotations
        def process_originals(img_list, split_name):
            for img_path in img_list:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = maybe_upscale(img, upscale_factor)
                h, w = img.shape[:2]
                
                new_name = img_path.name
                cv2.imwrite(str(output_dir / cls / new_name), img)
                cv2.imwrite(str(output_dir / 'images' / new_name), img)
                
                annotations[new_name] = {
                    "source_image": new_name,
                    "split": split_name,
                    "class_id": cls_id,
                    "class_name": cls,
                    "crop_size": [w, h],
                    "is_augmented": False
                }
                
        process_originals(train_images, "train")
        process_originals(val_images, "val")
        
        # Generate Augmentations for train
        needed_train = train_target - len(train_images)
        if needed_train > 0 and len(train_images) > 0:
            for i in tqdm(range(needed_train), desc=f"Augmenting {cls} (train)"):
                src_path = random.choice(train_images)
                img = cv2.imread(str(src_path))
                if img is None: continue
                
                aug_img = augmenter(image=img)['image']
                aug_img = maybe_upscale(aug_img, upscale_factor)
                h, w = aug_img.shape[:2]
                base_name = src_path.stem
                aug_name = f"{base_name}_trainaug{i:04d}{src_path.suffix}"
                
                cv2.imwrite(str(output_dir / cls / aug_name), aug_img)
                cv2.imwrite(str(output_dir / 'images' / aug_name), aug_img)
                
                annotations[aug_name] = {
                    "source_image": src_path.name,
                    "split": "train",
                    "class_id": cls_id,
                    "class_name": cls,
                    "crop_size": [w, h],
                    "is_augmented": True
                }

        # Generate Augmentations for val
        needed_val = val_target - len(val_images)
        if needed_val > 0 and len(val_images) > 0:
            for i in tqdm(range(needed_val), desc=f"Augmenting {cls} (val)"):
                src_path = random.choice(val_images)
                img = cv2.imread(str(src_path))
                if img is None: continue
                
                aug_img = augmenter(image=img)['image']
                aug_img = maybe_upscale(aug_img, upscale_factor)
                h, w = aug_img.shape[:2]
                base_name = src_path.stem
                aug_name = f"{base_name}_valaug{i:04d}{src_path.suffix}"
                
                cv2.imwrite(str(output_dir / cls / aug_name), aug_img)
                cv2.imwrite(str(output_dir / 'images' / aug_name), aug_img)
                
                annotations[aug_name] = {
                    "source_image": src_path.name,
                    "split": "val",
                    "class_id": cls_id,
                    "class_name": cls,
                    "crop_size": [w, h],
                    "is_augmented": True
                }
                
    with open(output_dir / 'annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
        
    print(f"\nCreated {len(annotations)} total crops in {output_dir}")
    for split in ['train', 'val']:
        for cls in classes:
            count = sum(1 for v in annotations.values() if v['split'] == split and v['class_name'] == cls)
            print(f"  {split} - {cls}: {count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--train-target', type=int, default=800)
    parser.add_argument('--val-target', type=int, default=200)
    parser.add_argument('--upscale-factor', type=float, default=1.0,
                        help='Upscale factor for output images (e.g., 1.5 or 2.0). 1.0 keeps original size.')
    args = parser.parse_args()

    if args.upscale_factor <= 0:
        raise ValueError("--upscale-factor must be > 0")
    
    random.seed(42)
    np.random.seed(42)
    
    augment_and_save(args.input, args.output, args.train_target, args.val_target, args.upscale_factor)

if __name__ == '__main__':
    main()
