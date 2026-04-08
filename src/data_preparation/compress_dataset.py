"""
Compress images in a dataset folder (lossless or near-lossless).

Converts PNG images to optimized PNG (max compression) or to high-quality JPEG.
Useful for reducing disk usage of augmented datasets.

Usage:
    # Lossless PNG compression (re-compress PNGs with max compression)
    python src/data_preparation/compress_dataset.py --dir data/yolo_augmented --mode lossless
    
    # Near-lossless: convert PNGs to JPEG quality 95
    python src/data_preparation/compress_dataset.py --dir data/yolo_augmented --mode jpeg --quality 95

    # Convert to JPEG and force image size to match a reference train folder
    python src/data_preparation/compress_dataset.py --dir data/val-macro/yolo_single_aug --mode jpeg --quality 95 --reference-dir data/val-macro/yolo_single/images/train
"""

import argparse
import cv2
from pathlib import Path


def _find_reference_size(reference_dir: str):
    """Return (width, height) from the first readable image in reference_dir."""
    ref_path = Path(reference_dir)
    if not ref_path.exists():
        return None

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    for img_path in sorted(ref_path.rglob('*')):
        if img_path.suffix.lower() in image_extensions:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                return (w, h)
    return None


def _resize_if_needed(img, target_size):
    """Resize image to target_size=(w,h) only when needed."""
    if target_size is None:
        return img

    h, w = img.shape[:2]
    target_w, target_h = target_size
    if w == target_w and h == target_h:
        return img

    interp = cv2.INTER_CUBIC if (target_w > w or target_h > h) else cv2.INTER_AREA
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def compress_dataset(data_dir: str, mode: str = 'lossless', quality: int = 95, reference_dir: str = None):
    """
    Compress images in a dataset directory.
    
    Args:
        data_dir: Path to dataset (searches images/ subdirectories)
        mode: 'lossless' (optimized PNG) or 'jpeg' (convert to JPEG)
        quality: JPEG quality (only used when mode='jpeg')
        reference_dir: Optional folder to match image size from (first readable image).
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Input directory not found: {data_dir}")
        return

    target_size = None
    if reference_dir:
        target_size = _find_reference_size(reference_dir)
        if target_size is None:
            print(f"Could not read a reference image from: {reference_dir}")
            return
    
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
    if target_size:
        print(f"Resize target: {target_size[0]}x{target_size[1]} (from {reference_dir})")
    
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
                        img = _resize_if_needed(img, target_size)
                        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                        size_after = img_path.stat().st_size
                        total_after += size_after
                        converted += 1
                    else:
                        total_after += size_before
                else:
                    total_after += size_before  # skip non-PNGs in lossless mode
                    
            elif mode == 'jpeg':
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is not None:
                    img = _resize_if_needed(img, target_size)

                    # Keep .jpg extension for consistency
                    new_path = img_path if img_path.suffix.lower() == '.jpg' else img_path.with_suffix('.jpg')
                    cv2.imwrite(str(new_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])

                    # Remove original when extension changed
                    if new_path != img_path and img_path.exists():
                        img_path.unlink()

                    size_after = new_path.stat().st_size
                    total_after += size_after
                    converted += 1
                else:
                    total_after += size_before
            
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
    parser.add_argument('--reference-dir', type=str, default=None,
                        help='Optional reference image folder to enforce output image dimensions')
    
    args = parser.parse_args()
    compress_dataset(args.dir, args.mode, args.quality, args.reference_dir)
