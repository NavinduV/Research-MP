"""
Compress images in a dataset folder (lossless or near-lossless).

Converts PNG images to optimized PNG (max compression) or to high-quality JPEG.
Useful for reducing disk usage of augmented datasets.

Usage:
    # Lossless PNG compression (re-compress PNGs with max compression)
    python src/data_preparation/compress_dataset.py --dir data/yolo_augmented --mode lossless
    
    # Near-lossless: convert PNGs to JPEG quality 95
    python src/data_preparation/compress_dataset.py --dir data/yolo_augmented --mode jpeg --quality 95
"""

import argparse
import cv2
import os
from pathlib import Path


def compress_dataset(data_dir: str, mode: str = 'lossless', quality: int = 95):
    """
    Compress images in a dataset directory.
    
    Args:
        data_dir: Path to dataset (searches images/ subdirectories)
        mode: 'lossless' (optimized PNG) or 'jpeg' (convert to JPEG)
        quality: JPEG quality (only used when mode='jpeg')
    """
    data_path = Path(data_dir)
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_path.rglob(f'*{ext}'))
    
    if not image_files:
        print(f"No images found in {data_dir}")
        return
    
    print(f"Found {len(image_files)} images in {data_dir}")
    print(f"Mode: {mode}" + (f" (quality={quality})" if mode == 'jpeg' else ''))
    
    total_before = 0
    total_after = 0
    converted = 0
    errors = 0
    
    for idx, img_path in enumerate(sorted(image_files), 1):
        try:
            size_before = img_path.stat().st_size
            total_before += size_before
            
            if mode == 'lossless':
                # Re-save PNG with maximum compression
                if img_path.suffix.lower() == '.png':
                    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                        size_after = img_path.stat().st_size
                        total_after += size_after
                        converted += 1
                    else:
                        total_after += size_before
                else:
                    total_after += size_before  # skip non-PNGs in lossless mode
                    
            elif mode == 'jpeg':
                if img_path.suffix.lower() in {'.png', '.bmp', '.tiff'}:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        new_path = img_path.with_suffix('.jpg')
                        cv2.imwrite(str(new_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                        
                        # Update corresponding label file if name changed
                        # (label files use stem, so .txt name stays the same)
                        
                        # Remove original
                        img_path.unlink()
                        
                        size_after = new_path.stat().st_size
                        total_after += size_after
                        converted += 1
                    else:
                        total_after += size_before
                else:
                    total_after += size_before  # already JPEG
            
            if idx % 50 == 0 or idx == len(image_files):
                saved_mb = (total_before - total_after) / (1024 * 1024)
                print(f"  [{idx}/{len(image_files)}] Saved {saved_mb:.1f} MB so far...")
                
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
            errors += 1
            total_after += img_path.stat().st_size if img_path.exists() else 0
    
    before_mb = total_before / (1024 * 1024)
    after_mb = total_after / (1024 * 1024)
    saved_mb = before_mb - after_mb
    ratio = (after_mb / before_mb * 100) if before_mb > 0 else 100
    
    print(f"\n{'='*50}")
    print(f"✅ Compression complete")
    print(f"  Files processed: {converted}")
    print(f"  Before: {before_mb:.1f} MB")
    print(f"  After:  {after_mb:.1f} MB")
    print(f"  Saved:  {saved_mb:.1f} MB ({100-ratio:.0f}% reduction)")
    if errors:
        print(f"  Errors: {errors}")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compress dataset images')
    parser.add_argument('--dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--mode', type=str, choices=['lossless', 'jpeg'], default='lossless',
                        help='Compression mode (default: lossless)')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality 1-100 (default: 95, only used with --mode jpeg)')
    
    args = parser.parse_args()
    compress_dataset(args.dir, args.mode, args.quality)
