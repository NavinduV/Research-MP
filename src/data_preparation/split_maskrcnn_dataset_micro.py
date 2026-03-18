import argparse
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_dataset(input_dir: Path, output_dir: Path, split_ratio: float = 0.8):
    print(f"\n{'='*60}")
    print(f"SPLITTING MASK R-CNN DATASET: {input_dir.name}")
    print(f"{'='*60}")

    images_in = input_dir / "images"
    masks_in = input_dir / "masks"
    ann_file = input_dir / "annotations.json"

    if not ann_file.exists():
        print(f"ERROR: No annotations.json found in {input_dir}")
        return

    with open(ann_file, "r") as f:
        annotations = json.load(f)

    # Group by source image to prevent data leakage.
    groups = defaultdict(list)
    for img_name, ann in annotations.items():
        if 'source_image' in ann:
            src = ann['source_image']
        else:
            if '_gt' in img_name:
                src = img_name.split('_gt')[0]
            elif '_crop' in img_name:
                src = img_name.split('_crop')[0]
            else:
                src = img_name
        
        groups[src].append((img_name, ann))

    source_images = list(groups.keys())
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(source_images)

    train_cutoff = int(len(source_images) * split_ratio)
    train_sources = set(source_images[:train_cutoff])

    splits = {"train": {}, "val": {}}
    for s in splits.keys():
        (output_dir / s / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / s / "masks").mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count = 0

    print("[1/2] Copying images and masks into splits...")
    for src, items in groups.items():
        split_name = "train" if src in train_sources else "val"
        for img_name, ann in items:
            src_img_path = images_in / img_name
            if src_img_path.exists():
                shutil.copy(src_img_path, output_dir / split_name / "images" / img_name)
            
            mask_name = ann.get('mask_file')
            if mask_name:
                src_mask_path = masks_in / mask_name
                if src_mask_path.exists():
                    shutil.copy(src_mask_path, output_dir / split_name / "masks" / mask_name)

            splits[split_name][img_name] = ann

            if split_name == "train":
                train_count += 1
            else:
                val_count += 1

    print("[2/2] Saving split annotations.json...")
    with open(output_dir / "train" / "annotations.json", "w") as f:
        json.dump(splits["train"], f, indent=2)

    with open(output_dir / "val" / "annotations.json", "w") as f:
        json.dump(splits["val"], f, indent=2)

    print(f"\nCompleted! Total Original Crops : {train_count + val_count}")
    print(f"  -> Train: {train_count} crops (from {len(train_sources)} base images)")
    print(f"  -> Val  : {val_count} crops (from {len(source_images) - len(train_sources)} base images)")
    print(f"Saved to {output_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default 0.8)")
    
    args = parser.parse_args()
    
    split_dataset(Path(args.input), Path(args.output), args.ratio)
