"""
Convert Label Studio exports to various formats needed by different models.

Supports conversions:
- Label Studio JSON -> YOLO format (for YOLOv8 training)
- Label Studio JSON -> COCO format (for Mask R-CNN training)
- Extract patches for EfficientNet classification

Usage:
    # Convert to YOLO format
    python src/convert_labels.py --input data/labelstudio_export.json --output data/yolo --format yolo

    # Convert to COCO format  
    python src/convert_labels.py --input data/labelstudio_export.json --output data/annotations --format coco

    # Extract classification patches
    python src/convert_labels.py --input data/labelstudio_export.json --output data/patches --format patches
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


def main():
    parser = argparse.ArgumentParser(description='Convert Label Studio exports')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to Label Studio JSON export')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--format', type=str, required=True,
                        choices=['yolo', 'coco', 'patches'],
                        help='Output format')
    parser.add_argument('--images', type=str, default='data/stitched',
                        help='Directory containing source images')
    parser.add_argument('--patch-size', type=int, default=224,
                        help='Patch size for classification (default: 224)')
    
    args = parser.parse_args()
    
    # Load data
    data = load_labelstudio_export(args.input)
    
    if args.format == 'yolo':
        convert_to_yolo(data, args.output, args.images)
    elif args.format == 'coco':
        convert_to_coco(data, args.output, args.images)
    elif args.format == 'patches':
        extract_patches(data, args.output, args.images, args.patch_size)


if __name__ == "__main__":
    main()
