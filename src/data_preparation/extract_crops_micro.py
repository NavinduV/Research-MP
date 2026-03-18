import argparse
import os
import json
from pathlib import Path
from collections import defaultdict

import cv2
from tqdm import tqdm

DEFAULT_CLASS_NAMES = {0: 'fiber', 1: 'film', 2: 'fragment'}

def extract_crops_micro(yolo_dir: str, output_dir: str, padding: int = 15,
                        min_crop_size: int = 8):
    """
    Extract crops from a micro dataset containing images and YOLO labels,
    without split directories (train/val/test).

    Output folder structure follows:
    - output_dir/
      - images/ (all crops)
      - fiber/ (crops of fibers)
      - film/ (crops of films)
      - fragment/ (crops of fragments)
      - annotations.json (dictionary of crop filenames)
    """
    yolo_dir = Path(yolo_dir)
    output_dir = Path(output_dir)
    
    class_names = DEFAULT_CLASS_NAMES
    
    img_dir = yolo_dir / 'images'
    lbl_dir = yolo_dir / 'labels'
    
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Images or labels dir missing in {yolo_dir}")
        return
        
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    for cls_name in class_names.values():
        (output_dir / cls_name).mkdir(parents=True, exist_ok=True)
        
    img_files = sorted(
        [f for f in img_dir.iterdir()
         if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
    )
    
    if not img_files:
        print(f"No images found in {img_dir}")
        return
        
    annotations = {}
        
    for img_path in tqdm(img_files, desc="Extracting crops"):
        lbl_path = lbl_dir / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        with open(lbl_path) as f:
            lines = f.read().strip().splitlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls_id = int(float(parts[0]))
            cx_n, cy_n, bw_n, bh_n = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            cx, cy = cx_n * w, cy_n * h
            bw, bh = bw_n * w, bh_n * h
            
            x1 = int(cx - bw / 2 - padding)
            y1 = int(cy - bh / 2 - padding)
            x2 = int(cx + bw / 2 + padding)
            y2 = int(cy + bh / 2 + padding)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            if crop_w < min_crop_size or crop_h < min_crop_size:
                continue
            
            crop = img[y1:y2, x1:x2]
            
            cls_name = class_names.get(cls_id, f'class_{cls_id}')
            crop_filename = f"{img_path.stem}_gt{i:04d}_{cls_name}.png"
            
            crop_path_cls = output_dir / cls_name / crop_filename
            crop_path_img = output_dir / 'images' / crop_filename
            
            cv2.imwrite(str(crop_path_cls), crop)
            cv2.imwrite(str(crop_path_img), crop)
            
            annotations[crop_filename] = {}

    with open(output_dir / 'annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
        
    print(f"\nExtraction complete to {output_dir}")
    print(f"Total crops: {len(annotations)}")

def main():
    parser = argparse.ArgumentParser(
        description='Extract crops from a flat YOLO dataset (images/ and labels/ directories)')
    parser.add_argument('--yolo-dir', type=str, required=True,
                        help='Path to YOLO dataset (with images/ and labels/ directly inside)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for crops')
    parser.add_argument('--padding', type=int, default=15,
                        help='Pixels of padding around each crop (default: 15)')
    
    args = parser.parse_args()
    extract_crops_micro(
        yolo_dir=args.yolo_dir,
        output_dir=args.output,
        padding=args.padding
    )

if __name__ == '__main__':
    main()
