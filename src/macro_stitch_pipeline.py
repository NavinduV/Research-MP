import os
import cv2
import json
import glob
import numpy as np
from typing import List, Tuple

# Disable OpenCL globally to avoid GPU-related errors during stitching
cv2.ocl.setUseOpenCL(False)


def load_images_from_folder(folder_path: str, exts: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')) -> List[np.ndarray]:
    files = sorted([p for p in glob.glob(os.path.join(folder_path, '*')) if p.lower().endswith(exts)])
    imgs = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        imgs.append(img)
    return imgs


def estimate_background(img: np.ndarray, blur_ksize: int = 101) -> np.ndarray:
    """Estimate low-frequency background using large Gaussian blur on grayscale."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    background = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return background.astype(np.float32)


def correct_illumination(img: np.ndarray, blur_ksize: int = 101, eps: float = 1e-6) -> np.ndarray:
    """Correct uneven illumination by dividing by a blurred background estimate.

    Operates on each channel using the grayscale background estimate to preserve
    color balance while removing shading gradients.
    """
    img_f = img.astype(np.float32)
    background = estimate_background(img, blur_ksize=blur_ksize)
    mean_bg = background.mean() if background.size else 1.0
    background = np.maximum(background, eps)

    out = np.empty_like(img_f)
    for c in range(3):
        out[:, :, c] = img_f[:, :, c] / background * mean_bg

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def enhance_contrast_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Enhance contrast safely using CLAHE on the L channel in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return img2


def stitch_images(image_list: List[np.ndarray], max_stitch_dim: int = 2048) -> np.ndarray:
    """Stitch a list of images using OpenCV's Stitcher (classical feature-based).

    To avoid excessive memory allocations (OpenCV's stitcher may try to create
    very large intermediate images), this function will downscale tiles so the
    maximum tile dimension is no larger than `max_stitch_dim`. The stitched
    panorama will therefore be at the downscaled resolution; the caller gets
    back the produced panorama and should consult metadata for the scale.

    This keeps the pipeline classical (no ML). For very challenging cases a
    more advanced blending/exposure compensation (opencv-contrib) may be used.
    """
    if not image_list:
        raise ValueError("No images to stitch")
    if len(image_list) == 1:
        return image_list[0]

    # We'll attempt stitching and, on memory errors or other failures that
    # indicate huge intermediate images, retry with progressively smaller
    # max_stitch_dim values. This avoids OOM crashes like allocating TBs.
    attempt_dim = int(max_stitch_dim)
    min_dim = 256
    last_err = None

    while attempt_dim >= min_dim:
        # compute max dim among tiles and resize if needed
        max_tile_dim = max(max(im.shape[0], im.shape[1]) for im in image_list)
        scale = 1.0
        if max_tile_dim > attempt_dim:
            scale = float(attempt_dim) / float(max_tile_dim)

        resized = [
            cv2.resize(
                im,
                (max(1, int(im.shape[1] * scale)), max(1, int(im.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            if scale != 1.0 else im
            for im in image_list
        ]

        # Log dimensions for debugging
        print(f"Attempting stitching with attempt_dim={attempt_dim}, scale={scale}")
        print(f"Resized image dimensions: {[im.shape for im in resized]}")

        # Attempt to use the available Stitcher API variant
        try:
            try:
                stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
            except Exception:
                stitcher = cv2.createStitcher(False)

            status, pano = stitcher.stitch(resized)
            if status != cv2.Stitcher_OK:
                raise RuntimeError(f"Stitcher failed with status {status}")

            # success
            return pano

        except cv2.error as e:
            last_err = e
            # If error mentions insufficient memory or allocation, retry smaller
            msg = str(e)
            if 'Insufficient memory' in msg or 'Failed to allocate' in msg or 'OutOfMemory' in msg:
                # reduce attempt_dim more aggressively and retry
                attempt_dim = max(attempt_dim // 2, min_dim)
                print(f"OpenCV stitcher OOM at dim {attempt_dim * 2}, retrying with {attempt_dim}...")
                continue
            else:
                # other cv2 error: re-raise
                raise

    # If we exit loop without success, raise the last encountered error
    raise RuntimeError(
        f"Stitching failed after retries; last error: {last_err}. Final attempt_dim={attempt_dim}, image dimensions={[im.shape for im in image_list]}"
    )


def prepare_for_yolo(img: np.ndarray, output_image_path: str, max_dim: int = 2048) -> dict:
    """Resize (if needed) and save image; produce a small meta JSON for labeling.

    Creates an empty YOLO label file next to the image so labeling GUIs know
    the image exists and can open it. Returns metadata including scale factors.
    """
    h, w = img.shape[:2]
    scale = 1.0
    # Ensure aspect ratio is preserved during resizing
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)  # Use area interpolation for downscaling
    else:
        img_out = img.copy()

    # Save intermediate results for debugging
    debug_dir = os.path.join(os.path.dirname(output_image_path), 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, 'resized_image.png'), img_out)  # Save resized image for inspection

    os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)
    cv2.imwrite(output_image_path, img_out)

    # create empty YOLO label file
    label_path = os.path.splitext(output_image_path)[0] + '.txt'
    if not os.path.exists(label_path):
        open(label_path, 'a').close()

    meta = {
        'original_size': [int(w), int(h)],
        'output_size': [int(img_out.shape[1]), int(img_out.shape[0])],
        'scale': float(scale),
        'image_path': output_image_path,
        'label_path': label_path,
    }
    with open(os.path.splitext(output_image_path)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def process_folder(input_folder: str, output_image_path: str, *,
                   illum_blur: int = 101, clahe_clip: float = 2.0, max_dim: int = 2048,
                   max_stitch_dim: int = 2048) -> dict:
    """Full pipeline: load, correct illum, enhance contrast, stitch, and save for YOLO."""
    print(f"Loading images from {input_folder}")
    imgs = load_images_from_folder(input_folder)
    if not imgs:
        raise RuntimeError("No images found in folder")

    print(f"Found {len(imgs)} images — applying illumination correction and CLAHE")
    preproc = []
    for im in imgs:
        im1 = correct_illumination(im, blur_ksize=illum_blur)
        im2 = enhance_contrast_clahe(im1, clip_limit=clahe_clip)
        preproc.append(im2)

    print("Stitching images...")
    pano = stitch_images(preproc, max_stitch_dim=max_stitch_dim)

    print(f"Preparing YOLO-ready output: {output_image_path}")
    meta = prepare_for_yolo(pano, output_image_path, max_dim=max_dim)
    print("Done — output saved and empty YOLO label file created if none existed.")
    return meta


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Macro image stitching pipeline (classical OpenCV).')
    parser.add_argument('input_folder', help='Folder with overlapping macro image tiles')
    parser.add_argument('output_image', help='Path for stitched output image (e.g. data/stitched/stitched.png)')
    parser.add_argument('--illum-blur', type=int, default=101, help='Kernel size for background blur (odd int)')
    parser.add_argument('--clahe-clip', type=float, default=2.0, help='CLAHE clip limit')
    parser.add_argument('--max-dim', type=int, default=2048, help='Max dimension for output image (resizing)')

    args = parser.parse_args()
    meta = process_folder(args.input_folder, args.output_image, illum_blur=args.illum_blur, clahe_clip=args.clahe_clip, max_dim=args.max_dim)
    print(json.dumps(meta, indent=2))
