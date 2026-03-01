"""
Convert Label Studio exports to various formats needed by different models.

Supports conversions:
- Label Studio JSON -> YOLO format (for YOLOv8 training)
- Label Studio JSON -> COCO format (for Mask R-CNN training)
- Extract patches for EfficientNet classification

Usage:
    # Convert to YOLO format
    python src/data_preparation/convert_labels.py --input data/labelstudio_export.json --output data/yolo --format yolo

    # Convert to COCO format  
    python src/data_preparation/convert_labels.py --input data/labelstudio_export.json --output data/annotations --format coco

    # Extract classification patches
    python src/data_preparation/convert_labels.py --input data/labelstudio_export.json --output data/patches --format patches
"""

import json
import os
import shutil
from pathlib import Path
import argparse
import cv2
import numpy as np
from typing import Dict, List, Tuple


# Class mapping
CLASS_MAP = {
    'fiber': 0,
    'film': 1,
    'fragment': 2,
    'Fiber': 0,
    'Film': 1,
    'Fragment': 2
}


def load_labelstudio_export(json_path: str) -> List[Dict]:
    """Load Label Studio JSON export."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} annotated images")
    return data


def get_image_path_from_ls(ls_path: str, images_dir: str) -> str:
    """
    Convert Label Studio image path to actual file path.
    Label Studio paths are like: /data/local-files/?d=images/sample1.png
    """
    if '?d=' in ls_path:
        # Local file path
        filename = ls_path.split('?d=')[-1]
    elif ls_path.startswith('/data/upload/'):
        # Uploaded file
        filename = ls_path.split('/')[-1]
    else:
        filename = Path(ls_path).name
    
    # Search for file in images directory
    for root, dirs, files in os.walk(images_dir):
        if filename in files:
            return os.path.join(root, filename)
    
    return os.path.join(images_dir, filename)


def convert_to_yolo(data: List[Dict], output_dir: str, images_source: str, 
                    train_split: float = 0.8):
    """
    Convert Label Studio annotations to YOLO format.
    
    YOLO format: class_id center_x center_y width height (normalized 0-1)
    """
    output_path = Path(output_dir)
    
    # Create directories
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split data
    np.random.shuffle(data)
    split_idx = int(len(data) * train_split)
    splits = {
        'train': data[:split_idx],
        'val': data[split_idx:]
    }
    
    for split_name, split_data in splits.items():
        for item in split_data:
            if 'annotations' not in item or not item['annotations']:
                continue
            
            # Get image info
            img_path = item.get('data', {}).get('image', '')
            actual_path = get_image_path_from_ls(img_path, images_source)
            
            if not os.path.exists(actual_path):
                print(f"Warning: Image not found: {actual_path}")
                continue
            
            # Read image to get dimensions
            img = cv2.imread(actual_path)
            if img is None:
                print(f"Warning: Could not read: {actual_path}")
                continue
            
            img_h, img_w = img.shape[:2]
            filename = Path(actual_path).stem
            
            # Copy image
            dest_img = output_path / 'images' / split_name / (filename + '.jpg')
            if actual_path.endswith('.png'):
                cv2.imwrite(str(dest_img), img)
            else:
                shutil.copy(actual_path, dest_img)
            
            # Process annotations
            labels = []
            for annotation in item['annotations']:
                for result in annotation.get('result', []):
                    if result['type'] != 'rectanglelabels':
                        continue
                    
                    value = result['value']
                    label = value['rectanglelabels'][0]
                    
                    if label not in CLASS_MAP:
                        print(f"Warning: Unknown class '{label}'")
                        continue
                    
                    class_id = CLASS_MAP[label]
                    
                    # Convert from percentage to normalized coordinates
                    x = value['x'] / 100
                    y = value['y'] / 100
                    w = value['width'] / 100
                    h = value['height'] / 100
                    
                    # YOLO format: center_x, center_y, width, height
                    cx = x + w / 2
                    cy = y + h / 2
                    
                    labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            # Save labels
            label_file = output_path / 'labels' / split_name / (filename + '.txt')
            with open(label_file, 'w') as f:
                f.write('\n'.join(labels))
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'fiber', 1: 'film', 2: 'fragment'},
        'nc': 3
    }
    
    import yaml
    with open(output_path / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"\nYOLO dataset created:")
    print(f"  Train images: {len(splits['train'])}")
    print(f"  Val images: {len(splits['val'])}")
    print(f"  Config: {output_path / 'dataset.yaml'}")


def convert_to_coco(data: List[Dict], output_dir: str, images_source: str):
    """
    Convert Label Studio annotations to COCO format for Mask R-CNN.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    coco = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 1, 'name': 'fiber'},
            {'id': 2, 'name': 'film'},
            {'id': 3, 'name': 'fragment'}
        ]
    }
    
    img_id = 0
    ann_id = 0
    
    for item in data:
        if 'annotations' not in item or not item['annotations']:
            continue
        
        # Get image info
        img_path = item.get('data', {}).get('image', '')
        actual_path = get_image_path_from_ls(img_path, images_source)
        
        if not os.path.exists(actual_path):
            continue
        
        img = cv2.imread(actual_path)
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]
        filename = Path(actual_path).name
        
        # Add image entry
        coco['images'].append({
            'id': img_id,
            'file_name': filename,
            'width': img_w,
            'height': img_h
        })
        
        # Process annotations
        for annotation in item['annotations']:
            for result in annotation.get('result', []):
                value = result['value']
                
                # Handle both rectangles and polygons
                if result['type'] == 'rectanglelabels':
                    label = value['rectanglelabels'][0]
                    if label not in CLASS_MAP:
                        continue
                    
                    # Convert percentage to pixels
                    x = value['x'] / 100 * img_w
                    y = value['y'] / 100 * img_h
                    w = value['width'] / 100 * img_w
                    h = value['height'] / 100 * img_h
                    
                    # Create polygon from rectangle
                    segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]
                    bbox = [x, y, w, h]
                    area = w * h
                    
                elif result['type'] == 'polygonlabels':
                    label = value['polygonlabels'][0]
                    if label not in CLASS_MAP:
                        continue
                    
                    # Convert polygon points
                    points = value['points']
                    poly = []
                    xs, ys = [], []
                    for pt in points:
                        px = pt[0] / 100 * img_w
                        py = pt[1] / 100 * img_h
                        poly.extend([px, py])
                        xs.append(px)
                        ys.append(py)
                    
                    segmentation = [poly]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    area = 0.5 * abs(sum(xs[i]*ys[i+1] - xs[i+1]*ys[i] for i in range(-1, len(xs)-1)))
                else:
                    continue
                
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': CLASS_MAP[label] + 1,  # COCO uses 1-indexed
                    'segmentation': segmentation,
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': 0
                })
                ann_id += 1
        
        img_id += 1
    
    # Save COCO JSON
    with open(output_path / 'annotations.json', 'w') as f:
        json.dump(coco, f, indent=2)
    
    print(f"\nCOCO dataset created:")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Output: {output_path / 'annotations.json'}")


def extract_patches(data: List[Dict], output_dir: str, images_source: str, 
                    patch_size: int = 224):
    """
    Extract image patches for EfficientNet classification training.
    Each patch is saved in a class-specific folder.
    """
    output_path = Path(output_dir)
    
    # Create class directories
    for class_name in ['fiber', 'film', 'fragment']:
        (output_path / class_name).mkdir(parents=True, exist_ok=True)
    
    patch_counts = {'fiber': 0, 'film': 0, 'fragment': 0}
    
    for item in data:
        if 'annotations' not in item or not item['annotations']:
            continue
        
        img_path = item.get('data', {}).get('image', '')
        actual_path = get_image_path_from_ls(img_path, images_source)
        
        if not os.path.exists(actual_path):
            continue
        
        img = cv2.imread(actual_path)
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]
        
        for annotation in item['annotations']:
            for result in annotation.get('result', []):
                if result['type'] != 'rectanglelabels':
                    continue
                
                value = result['value']
                label = value['rectanglelabels'][0].lower()
                
                if label not in patch_counts:
                    continue
                
                # Get bounding box
                x = int(value['x'] / 100 * img_w)
                y = int(value['y'] / 100 * img_h)
                w = int(value['width'] / 100 * img_w)
                h = int(value['height'] / 100 * img_h)
                
                # Ensure valid bounds
                x = max(0, x)
                y = max(0, y)
                x2 = min(img_w, x + w)
                y2 = min(img_h, y + h)
                
                # Extract patch
                patch = img[y:y2, x:x2]
                
                if patch.size == 0:
                    continue
                
                # Resize to standard size
                patch = cv2.resize(patch, (patch_size, patch_size))
                
                # Save patch
                patch_name = f"{label}_{patch_counts[label]:05d}.jpg"
                cv2.imwrite(str(output_path / label / patch_name), patch)
                patch_counts[label] += 1
    
    print(f"\nPatches extracted:")
    for class_name, count in patch_counts.items():
        print(f"  {class_name}: {count}")
    print(f"  Output: {output_path}")


def coco_export_to_training_masks(export_dir: str, output_dir: str, mp_type: str) -> None:
    """
    Convert a Label Studio 'COCO with Images' export into the images/ + masks/
    folder structure expected by the Mask R-CNN training pipeline.

    Label Studio COCO export layout:
        export_dir/
            result.json      <- COCO annotations (polygons / RLE)
            images/
                *.png/*.jpg  <- original crop images

    Output layout:
        output_dir/          e.g. data/crops_fiber_sam/
            images/          <- copies of source images
            masks/           <- binary PNG mask per image (255=object, 0=bg)
            annotations.json <- {filename: {mask_file, class_name, class_id}}

    Usage:
        python src/data_preparation/convert_labels.py \\
            --format masks \\
            --input  exports/fiber_coco \\
            --output data/crops_fiber_sam \\
            --type   fiber
    """
    import shutil

    exp_path = Path(export_dir)
    out_path = Path(output_dir)

    # Locate COCO JSON
    result_json = None
    for name in ("result.json", "annotations.json", "coco.json"):
        if (exp_path / name).exists():
            result_json = exp_path / name
            break
    if result_json is None:
        print(f"ERROR: No COCO JSON found in {exp_path}")
        return

    with open(result_json) as f:
        coco = json.load(f)

    images_out = out_path / "images"
    masks_out  = out_path / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    id2img  = {img["id"]: img for img in coco["images"]}
    id2anns: dict = {}
    for ann in coco.get("annotations", []):
        id2anns.setdefault(ann["image_id"], []).append(ann)

    type_to_class_id = {"fiber": 0, "film": 1, "fragment": 2}
    ann_lookup = {}
    processed = skipped = 0

    for img_info in coco["images"]:
        img_id   = img_info["id"]
        filename = Path(img_info["file_name"]).name

        # Find source image
        src = next((p for p in [exp_path / "images" / filename, exp_path / filename]
                    if p.exists()), None)
        if src is None:
            print(f"  SKIP (not found): {filename}")
            skipped += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            print(f"  SKIP (unreadable): {filename}")
            skipped += 1
            continue
        h, w = img.shape[:2]

        # Rasterise all polygon segmentations into one binary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in id2anns.get(img_id, []):
            for poly in ann.get("segmentation", []):
                if len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
            if not ann.get("segmentation"):
                bx, by, bw, bh = [int(v) for v in ann.get("bbox", [0, 0, w, h])]
                cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)

        # Copy image
        shutil.copy2(src, images_out / filename)

        # Save mask
        stem = Path(filename).stem
        mask_filename = f"{stem}_mask.png"
        cv2.imwrite(str(masks_out / mask_filename), mask)

        ann_lookup[filename] = {
            "mask_file":  mask_filename,
            "class_name": mp_type,
            "class_id":   type_to_class_id.get(mp_type, 0),
        }
        processed += 1

    with open(out_path / "annotations.json", "w") as f:
        json.dump(ann_lookup, f, indent=2)

    print(f"\n✅ {mp_type} — {processed} masks created, {skipped} skipped")
    print(f"   images/ → {images_out}")
    print(f"   masks/  → {masks_out}")


def brush_png_to_training(json_export: str, masks_dir: str, output_dir: str,
                           mp_type: str, crops_source: str) -> None:
    """
    Convert a Label Studio 'Brush labels to PNG' export into the images/ + masks/
    folder structure expected by the Mask R-CNN training pipeline.

    How to export from Label Studio:
        1. Export → JSON  →  save as  exports/<type>/annotations.json
        2. Export → Brush labels to PNG  →  unzip into  exports/<type>/masks/

    The brush mask PNGs are named:
        {task_id}-annotation_{ann_id}-by-{user}-tag-{label}-{index}.png
    This function matches them back to original image filenames via the JSON export.

    Args:
        json_export:   Path to the Label Studio JSON export file
        masks_dir:     Folder containing the unzipped brush PNG mask files
        output_dir:    Output root  (e.g. data/crops_fiber_sam)
        mp_type:       'fiber' | 'film' | 'fragment'
        crops_source:  Folder that contains the original crop images
                       (e.g. data/crops/fiber)

    Output:
        output_dir/
            images/          <- original crop images
            masks/           <- binary PNG masks (255=object, 0=bg)
            annotations.json <- {filename: {mask_file, class_name, class_id}}

    Usage:
        python src/data_preparation/convert_labels.py \\
            --format brush_png \\
            --input  exports/fiber/annotations.json \\
            --masks  exports/fiber/masks \\
            --output data/crops_fiber_sam \\
            --images data/crops/fiber \\
            --type   fiber
    """
    json_path    = Path(json_export)
    masks_path   = Path(masks_dir)
    out_path     = Path(output_dir)
    crops_path   = Path(crops_source)

    if not json_path.exists():
        print(f"ERROR: JSON export not found: {json_path}")
        return
    if not masks_path.exists():
        print(f"ERROR: masks dir not found: {masks_path}")
        return

    with open(json_path) as f:
        tasks = json.load(f)

    images_out = out_path / "images"
    masks_out  = out_path / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    # Index all mask PNGs by their task_id prefix
    # Filename format: {task_id}-annotation_{ann_id}-by-{user}-tag-{label}-{index}.png
    mask_files = list(masks_path.glob("*.png"))
    task_id_to_masks: dict[int, list[Path]] = {}
    for mf in mask_files:
        try:
            task_id = int(mf.name.split("-")[0])
            task_id_to_masks.setdefault(task_id, []).append(mf)
        except ValueError:
            pass

    type_to_class_id = {"fiber": 0, "film": 1, "fragment": 2}
    ann_lookup = {}
    processed = skipped = 0

    for task in tasks:
        task_id = task["id"]

        # Extract original image filename from the task data
        img_url = task.get("data", {}).get("image", "")
        # Handles: /data/local-files/?d=fiber/crop.png  or  /data/upload/.../crop.png
        orig_name = Path(img_url.split("?d=")[-1]).name if "?d=" in img_url else Path(img_url).name
        if not orig_name:
            print(f"  SKIP task {task_id}: cannot parse image name from '{img_url}'")
            skipped += 1
            continue

        # Find source image
        src = next((p for p in [crops_path / orig_name,
                                 crops_path.parent / orig_name]
                    if p.exists()), None)
        if src is None:
            print(f"  SKIP task {task_id}: source image not found ({orig_name})")
            skipped += 1
            continue

        # Get mask PNGs for this task
        task_masks = task_id_to_masks.get(task_id, [])
        if not task_masks:
            print(f"  WARN task {task_id}: no brush mask found for {orig_name}")
            # Still copy the image so it at least appears in the dataset
        
        # Read image dimensions
        img = cv2.imread(str(src))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        # Merge all brush mask layers for this task into one binary mask
        merged_mask = np.zeros((h, w), dtype=np.uint8)
        for mf in task_masks:
            m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            merged_mask = np.maximum(merged_mask, m)

        # Binarise
        _, merged_mask = cv2.threshold(merged_mask, 127, 255, cv2.THRESH_BINARY)

        # Copy image
        shutil.copy2(src, images_out / orig_name)

        # Save mask
        stem = Path(orig_name).stem
        mask_filename = f"{stem}_mask.png"
        cv2.imwrite(str(masks_out / mask_filename), merged_mask)

        ann_lookup[orig_name] = {
            "mask_file":  mask_filename,
            "class_name": mp_type,
            "class_id":   type_to_class_id.get(mp_type, 0),
        }
        processed += 1

    with open(out_path / "annotations.json", "w") as f:
        json.dump(ann_lookup, f, indent=2)

    print(f"\n✅ {mp_type} — {processed} tasks converted, {skipped} skipped")
    print(f"   images/ → {images_out}")
    print(f"   masks/  → {masks_out}")
    print(f"   annotations → {out_path / 'annotations.json'}")


def main():
    parser = argparse.ArgumentParser(description='Convert Label Studio exports')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to Label Studio JSON export file (or COCO export folder for --format masks)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory (e.g. data/crops_fiber_sam)')
    parser.add_argument('--format', type=str, required=True,
                        choices=['yolo', 'coco', 'patches', 'masks', 'brush_png'],
                        help=(
                            'Output format. '
                            '"brush_png" = Brush labels to PNG export + JSON → images/+masks/. '
                            '"masks"     = COCO with Images export → images/+masks/.'
                        ))
    parser.add_argument('--images', type=str, default='data/stitched',
                        help='Source images directory (all formats; for brush_png use the crops/<type> folder)')
    parser.add_argument('--masks', type=str, default=None,
                        help='Folder containing unzipped Brush PNG masks (required for --format brush_png)')
    parser.add_argument('--patch-size', type=int, default=224,
                        help='Patch size for classification (default: 224)')
    parser.add_argument('--type', type=str, choices=['fiber', 'film', 'fragment'],
                        help='Microplastic type (required for --format brush_png or masks)')

    args = parser.parse_args()

    if args.format == 'brush_png':
        if not args.type:
            parser.error("--type is required for --format brush_png")
        if not args.masks:
            parser.error("--masks (unzipped brush PNG folder) is required for --format brush_png")
        brush_png_to_training(args.input, args.masks, args.output, args.type, args.images)
        return

    if args.format == 'masks':
        if not args.type:
            parser.error("--type is required when --format masks")
        coco_export_to_training_masks(args.input, args.output, args.type)
        return

    # Load data (LS JSON export)
    data = load_labelstudio_export(args.input)

    if args.format == 'yolo':
        convert_to_yolo(data, args.output, args.images)
    elif args.format == 'coco':
        convert_to_coco(data, args.output, args.images)
    elif args.format == 'patches':
        extract_patches(data, args.output, args.images, args.patch_size)


if __name__ == "__main__":
    main()
