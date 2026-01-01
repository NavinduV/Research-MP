import os
import cv2
import json
import glob
import numpy as np
from typing import List, Tuple, Optional, Dict

# Disable OpenCL globally to avoid GPU-related errors during stitching
cv2.ocl.setUseOpenCL(False)


def calculate_image_brightness(img_path: str) -> float:
    """Calculate the average brightness of an image.
    
    Args:
        img_path: Path to the image file
    
    Returns:
        Average brightness value (0-255)
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0.0
    
    # Convert to grayscale and calculate mean brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness


def group_images_by_brightness(folder_path: str, tolerance: int = 15, 
                               exts: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')) -> Dict[str, List[Dict]]:
    """Load images from folder and group them by brightness level.
    
    Args:
        folder_path: Path to folder containing images
        tolerance: Brightness tolerance for grouping (default 15)
        exts: Image file extensions
    
    Returns:
        Dictionary with brightness groups as keys and list of image info as values
    """
    files = sorted([p for p in glob.glob(os.path.join(folder_path, '*')) if p.lower().endswith(exts)])
    
    image_info = []
    for filepath in files:
        brightness = calculate_image_brightness(filepath)
        image_info.append({
            'path': filepath,
            'filename': os.path.basename(filepath),
            'brightness': brightness
        })
    
    # Group by brightness with tolerance
    brightness_groups = {}
    for img_info in image_info:
        brightness = img_info['brightness']
        
        # Find existing group within tolerance
        found_group = False
        for group_key in brightness_groups:
            if abs(brightness - group_key) <= tolerance:
                brightness_groups[group_key].append(img_info)
                found_group = True
                break
        
        # Create new group if not found
        if not found_group:
            brightness_groups[brightness] = [img_info]
    
    # Sort groups by brightness key
    sorted_groups = {}
    for key in sorted(brightness_groups.keys()):
        brightness_level = round(key)
        sorted_groups[brightness_level] = brightness_groups[key]
    
    return sorted_groups


def load_images_from_folder(folder_path: str, exts: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')) -> List[np.ndarray]:
    files = sorted([p for p in glob.glob(os.path.join(folder_path, '*')) if p.lower().endswith(exts)])
    imgs = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        imgs.append(img)
    return imgs


def upscale_image(img: np.ndarray, scale: float = 2.0, method: str = 'lanczos') -> np.ndarray:
    """Upscale an image by a given factor using high-quality interpolation.
    
    Args:
        img: Input image
        scale: Upscale factor (e.g., 2.0 = double size, 1.5 = 150%)
        method: Interpolation method ('lanczos', 'cubic', 'linear')
    
    Returns:
        Upscaled image
    """
    if scale <= 1.0:
        return img
    
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    interpolation_methods = {
        'lanczos': cv2.INTER_LANCZOS4,
        'cubic': cv2.INTER_CUBIC,
        'linear': cv2.INTER_LINEAR
    }
    
    interp = interpolation_methods.get(method.lower(), cv2.INTER_LANCZOS4)
    
    print(f"Upscaling image from {w}x{h} to {new_w}x{new_h} (scale={scale}x, method={method})")
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    return upscaled


def sharpen_image(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply unsharp masking to sharpen the image.
    
    Args:
        img: Input image
        strength: Sharpening strength (0.5 = subtle, 1.0 = normal, 2.0 = strong)
    
    Returns:
        Sharpened image
    """
    if strength <= 0:
        return img
    
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    
    # Unsharp mask: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    
    return sharpened


def denoise_image(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """Apply denoising to reduce noise while preserving edges.
    
    Args:
        img: Input image
        strength: Denoising strength (3-10 = light, 10-20 = medium, 20+ = strong)
    
    Returns:
        Denoised image
    """
    if strength <= 0:
        return img
    
    # Use Non-local Means Denoising (best quality for color images)
    # h = filter strength, hForColorComponents, templateWindowSize, searchWindowSize
    denoised = cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
    
    return denoised


def enhance_contrast(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Enhance contrast using CLAHE on the L channel in LAB color space.
    
    Args:
        img: Input image
        clip_limit: CLAHE clip limit (1.0-4.0 typical, higher = more contrast)
    
    Returns:
        Contrast-enhanced image
    """
    if clip_limit <= 0:
        return img
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def adjust_brightness_contrast(img: np.ndarray, brightness: int = 0, contrast: float = 1.0) -> np.ndarray:
    """Adjust brightness and contrast of the image.
    
    Args:
        img: Input image
        brightness: Brightness adjustment (-100 to 100, 0 = no change)
        contrast: Contrast multiplier (0.5 to 2.0, 1.0 = no change)
    
    Returns:
        Adjusted image
    """
    if brightness == 0 and contrast == 1.0:
        return img
    
    # Apply contrast and brightness: output = contrast * input + brightness
    adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    
    return adjusted


def auto_white_balance(img: np.ndarray) -> np.ndarray:
    """Apply automatic white balance correction using gray world assumption.
    
    Returns:
        White-balanced image
    """
    # Calculate mean of each channel
    b, g, r = cv2.split(img.astype(np.float32))
    
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    # Scale each channel
    if avg_b > 0:
        b = b * (avg_gray / avg_b)
    if avg_g > 0:
        g = g * (avg_gray / avg_g)
    if avg_r > 0:
        r = r * (avg_gray / avg_r)
    
    balanced = cv2.merge([
        np.clip(b, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(r, 0, 255).astype(np.uint8)
    ])
    
    return balanced


def enhance_image(img: np.ndarray, 
                  sharpen: float = 0.0,
                  denoise: int = 0,
                  contrast: float = 0.0,
                  brightness: int = 0,
                  auto_wb: bool = False) -> np.ndarray:
    """Apply multiple enhancement operations to improve image quality.
    
    Args:
        img: Input image
        sharpen: Sharpening strength (0 = off, 0.5-2.0 typical)
        denoise: Denoising strength (0 = off, 5-15 typical)
        contrast: CLAHE contrast enhancement (0 = off, 1.0-3.0 typical)
        brightness: Brightness adjustment (-100 to 100, 0 = no change)
        auto_wb: Apply automatic white balance
    
    Returns:
        Enhanced image
    """
    result = img.copy()
    
    # Apply enhancements in optimal order
    if auto_wb:
        print("  Applying auto white balance...")
        result = auto_white_balance(result)
    
    if denoise > 0:
        print(f"  Applying denoising (strength={denoise})...")
        result = denoise_image(result, strength=denoise)
    
    if contrast > 0:
        print(f"  Enhancing contrast (clip_limit={contrast})...")
        result = enhance_contrast(result, clip_limit=contrast)
    
    if brightness != 0:
        print(f"  Adjusting brightness ({brightness:+d})...")
        result = adjust_brightness_contrast(result, brightness=brightness)
    
    if sharpen > 0:
        print(f"  Sharpening (strength={sharpen})...")
        result = sharpen_image(result, strength=sharpen)
    
    return result


def detect_and_match_features(img1: np.ndarray, img2: np.ndarray, 
                               nfeatures: int = 10000, ratio_thresh: float = 0.75) -> Tuple[List, List, List]:
    """Detect SIFT features and match between two images using FLANN matcher with ratio test."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Use SIFT for better feature matching (more robust than ORB for this use case)
    try:
        sift = cv2.SIFT_create(nfeatures=nfeatures)
    except AttributeError:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    
    # Detect keypoints and compute descriptors
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)
    
    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        return [], [], []
    
    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find k=2 nearest neighbors for ratio test
    try:
        matches = flann.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return [], [], []
    
    # Apply Lowe's ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    
    return kp1, kp2, good_matches


def compute_homography(kp1: List, kp2: List, matches: List, 
                       min_matches: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute homography matrix from matched keypoints using RANSAC.
    
    Returns H such that points in img2 can be transformed to img1's coordinate system.
    i.e., img2 warped by H will align with img1.
    """
    if len(matches) < min_matches:
        return None, None
    
    # src_pts are from img1 (query), dst_pts are from img2 (train)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography: H transforms points from img2 to img1
    # So we want to find H such that src_pts = H * dst_pts
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    return H, mask


def stitch_pair(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Stitch img2 onto img1 using homography H (H transforms img2 coords to img1 coords).
    
    Uses high-quality interpolation and smart blending to preserve image quality.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Get corners of img2 in img1's coordinate system
    corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners_transformed = cv2.perspectiveTransform(corners_img2, H)
    
    # Corners of img1
    corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    
    # Find bounding box of combined images
    all_corners = np.concatenate([corners_img1, corners_transformed], axis=0)
    
    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))
    
    # Translation to keep everything in positive coordinates
    tx, ty = -x_min, -y_min
    translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    
    out_w = x_max - x_min
    out_h = y_max - y_min
    
    # Limit size to prevent memory issues (increased limit for quality)
    max_dim = 16000
    if out_w > max_dim or out_h > max_dim:
        scale = max_dim / max(out_w, out_h)
        out_w = int(out_w * scale)
        out_h = int(out_h * scale)
        scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
        translation = scale_mat @ translation
    
    # Use INTER_CUBIC for higher quality warping (better than default INTER_LINEAR)
    result = cv2.warpPerspective(img1, translation, (out_w, out_h), 
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    
    # Warp img2 using H combined with translation
    H_combined = translation @ H
    warped_img2 = cv2.warpPerspective(img2, H_combined, (out_w, out_h),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    
    # Create mask for img2 (also warped)
    mask2 = cv2.warpPerspective(np.ones((h2, w2), dtype=np.uint8) * 255, H_combined, (out_w, out_h))
    
    # Create mask for img1
    mask1 = cv2.warpPerspective(np.ones((h1, w1), dtype=np.uint8) * 255, translation, (out_w, out_h))
    
    # Non-overlap: where only img2 exists, use img2
    only_img2 = (mask1 == 0) & (mask2 > 0)
    result[only_img2] = warped_img2[only_img2]
    
    # Overlap region: use feather blending based on distance from edge
    overlap_mask = (mask1 > 0) & (mask2 > 0)
    if np.any(overlap_mask):
        # Create distance-based weights for smoother blending
        # Distance transform from edges of each mask
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
        
        # Normalize and create blend weights
        total_dist = dist1 + dist2 + 1e-6
        weight1 = dist1 / total_dist
        weight2 = dist2 / total_dist
        
        # Apply weighted blend only in overlap region
        for c in range(3):
            blended = (result[:, :, c].astype(np.float32) * weight1 + 
                      warped_img2[:, :, c].astype(np.float32) * weight2)
            result[:, :, c] = np.where(overlap_mask, blended.astype(np.uint8), result[:, :, c])
    
    return result


def stitch_images(image_list: List[np.ndarray], max_stitch_dim: int = 8192) -> np.ndarray:
    """Stitch a list of partial filter paper images using sequential pairwise stitching.
    
    This algorithm stitches images sequentially (1+2, then result+3, etc.) which works
    well for images captured in order with overlapping regions.
    
    Args:
        image_list: List of overlapping partial images of the filter paper
        max_stitch_dim: Maximum dimension for intermediate processing (default 8192 for high quality)
    
    Returns:
        Stitched mosaic image at full resolution
    """
    if not image_list:
        raise ValueError("No images to stitch")
    if len(image_list) == 1:
        return image_list[0]
    
    print(f"Stitching {len(image_list)} images using sequential pairwise stitching...")
    print(f"Quality mode: max_stitch_dim={max_stitch_dim}")
    
    # Check if we need to resize (only for very large images to prevent OOM)
    max_tile_dim = max(max(im.shape[0], im.shape[1]) for im in image_list)
    scale = 1.0
    if max_tile_dim > max_stitch_dim:
        scale = float(max_stitch_dim) / float(max_tile_dim)
        print(f"Resizing images from max dim {max_tile_dim} to {max_stitch_dim} (scale={scale:.3f})")
        working_images = [
            cv2.resize(im, (max(1, int(im.shape[1] * scale)), max(1, int(im.shape[0] * scale))),
                       interpolation=cv2.INTER_LANCZOS4)  # Use Lanczos for best quality downscaling
            for im in image_list
        ]
    else:
        # No resizing needed - work with original quality
        print(f"Working at full resolution (max dim: {max_tile_dim})")
        working_images = [im.copy() for im in image_list]
    
    print(f"Working image dimensions: {[im.shape[:2] for im in working_images]}")
    
    # Sequential pairwise stitching
    result = working_images[0]
    
    for i in range(1, len(working_images)):
        print(f"Stitching image {i+1}/{len(working_images)}...")
        
        next_img = working_images[i]
        
        # Find features and matches
        kp1, kp2, matches = detect_and_match_features(result, next_img)
        
        if len(matches) < 10:
            print(f"  Warning: Only {len(matches)} matches found, trying reverse direction...")
            # Try reverse direction
            kp2, kp1, matches = detect_and_match_features(next_img, result)
            if len(matches) < 10:
                print(f"  Error: Could not find enough matches for image {i+1}, skipping...")
                continue
            # Swap images for stitching
            H, mask = compute_homography(kp2, kp1, matches)
            if H is not None:
                result = stitch_pair(next_img, result, H)
            continue
        
        print(f"  Found {len(matches)} good matches")
        
        # Compute homography
        H, mask = compute_homography(kp1, kp2, matches)
        
        if H is None:
            print(f"  Error: Could not compute homography for image {i+1}, skipping...")
            continue
        
        # Check if homography is reasonable
        det = np.linalg.det(H[:2, :2])
        if not (0.1 < det < 10):
            print(f"  Warning: Homography has unusual determinant {det:.3f}, skipping...")
            continue
        
        # Stitch the pair
        result = stitch_pair(result, next_img, H)
        print(f"  Current mosaic size: {result.shape[:2]}")
    
    # Crop black borders
    result = crop_black_borders(result)
    
    print(f"Stitching complete. Final size: {result.shape[:2]}")
    return result


def crop_black_borders(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop black/dark borders from the stitched image."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find non-black regions
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    # Find bounding box of all non-black content
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    # Add small margin
    margin = 5
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(img.shape[1], x_max + margin)
    y_max = min(img.shape[0], y_max + margin)
    
    return img[y_min:y_max, x_min:x_max]


def prepare_for_yolo(img: np.ndarray, output_image_path: str, max_dim: int = 8192) -> dict:
    """Resize (if needed) and save image; produce a small meta JSON for labeling.

    Creates an empty YOLO label file next to the image so labeling GUIs know
    the image exists and can open it. Returns metadata including scale factors.
    
    Args:
        img: Input image
        output_image_path: Path to save the output image
        max_dim: Maximum dimension (default 8192 for high quality)
    """
    h, w = img.shape[:2]
    scale = 1.0
    
    # Only resize if image exceeds max_dim, preserving aspect ratio
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        # Use INTER_LANCZOS4 for best quality downscaling
        img_out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        print(f"Resized output from {w}x{h} to {new_w}x{new_h}")
    else:
        img_out = img.copy()
        print(f"Saving at full resolution: {w}x{h}")

    os.makedirs(os.path.dirname(output_image_path) or '.', exist_ok=True)
    
    # Save with maximum quality
    ext = os.path.splitext(output_image_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        # Use maximum JPEG quality
        cv2.imwrite(output_image_path, img_out, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        # Use lossless PNG (compression 0 = no compression, fastest but largest file)
        # compression 1 = minimal compression, good balance
        cv2.imwrite(output_image_path, img_out, [cv2.IMWRITE_PNG_COMPRESSION, 1])

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
                   max_dim: int = 8192, max_stitch_dim: int = 8192,
                   upscale: float = 1.0, upscale_method: str = 'lanczos',
                   sharpen: float = 0.0, denoise: int = 0, contrast: float = 0.0,
                   brightness: int = 0, auto_wb: bool = False) -> dict:
    """Full pipeline: load images, stitch, enhance, optionally upscale, and save.
    
    Args:
        input_folder: Folder containing overlapping partial images
        output_image_path: Path for the output stitched image
        max_dim: Maximum dimension for final output image (default 8192 for high quality)
        max_stitch_dim: Maximum dimension during stitching (default 8192 for high quality)
        upscale: Upscale factor for final image (1.0 = no upscaling, 2.0 = double size)
        upscale_method: Interpolation method for upscaling ('lanczos', 'cubic', 'linear')
        sharpen: Sharpening strength (0 = off, 0.5-2.0 typical)
        denoise: Denoising strength (0 = off, 5-15 typical)
        contrast: CLAHE contrast enhancement (0 = off, 1.0-3.0 typical)
        brightness: Brightness adjustment (-100 to 100, 0 = no change)
        auto_wb: Apply automatic white balance
    """
    print(f"Loading images from {input_folder}")
    imgs = load_images_from_folder(input_folder)
    if not imgs:
        raise RuntimeError("No images found in folder")

    print(f"Found {len(imgs)} images")

    print("Stitching images...")
    pano = stitch_images(imgs, max_stitch_dim=max_stitch_dim)
    
    # Apply enhancements if any are requested
    if sharpen > 0 or denoise > 0 or contrast > 0 or brightness != 0 or auto_wb:
        print("Enhancing image quality...")
        pano = enhance_image(pano, sharpen=sharpen, denoise=denoise, 
                            contrast=contrast, brightness=brightness, auto_wb=auto_wb)
    
    # Apply upscaling if requested
    if upscale > 1.0:
        pano = upscale_image(pano, scale=upscale, method=upscale_method)

    print(f"Saving output: {output_image_path}")
    meta = prepare_for_yolo(pano, output_image_path, max_dim=max_dim)
    print("Done — output saved and empty YOLO label file created if none existed.")
    return meta


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Macro image stitching pipeline for filter paper images.')
    parser.add_argument('input_folder', help='Folder with overlapping partial filter paper images')
    parser.add_argument('output_image', help='Path for stitched output image (e.g. data/stitched/stitched.png)')
    parser.add_argument('--max-dim', type=int, default=8192, help='Max dimension for output image (default: 8192)')
    parser.add_argument('--max-stitch-dim', type=int, default=8192, help='Max dimension during stitching (default: 8192)')
    parser.add_argument('--upscale', type=float, default=1.0, help='Upscale factor (default: 1.0, use 2.0 for 2x)')
    parser.add_argument('--upscale-method', type=str, default='lanczos', choices=['lanczos', 'cubic', 'linear'],
                        help='Upscaling interpolation method (default: lanczos)')
    
    # Enhancement options
    parser.add_argument('--sharpen', type=float, default=0.0, help='Sharpening strength (0=off, 0.5-2.0 typical)')
    parser.add_argument('--denoise', type=int, default=0, help='Denoising strength (0=off, 5-15 typical)')
    parser.add_argument('--contrast', type=float, default=0.0, help='Contrast enhancement (0=off, 1.0-3.0 typical)')
    parser.add_argument('--brightness', type=int, default=0, help='Brightness adjustment (-100 to 100)')
    parser.add_argument('--auto-wb', action='store_true', help='Apply automatic white balance')

    args = parser.parse_args()
    meta = process_folder(
        args.input_folder, 
        args.output_image, 
        max_dim=args.max_dim,
        max_stitch_dim=args.max_stitch_dim,
        upscale=args.upscale,
        upscale_method=args.upscale_method,
        sharpen=args.sharpen,
        denoise=args.denoise,
        contrast=args.contrast,
        brightness=args.brightness,
        auto_wb=args.auto_wb
    )
    print(json.dumps(meta, indent=2))
