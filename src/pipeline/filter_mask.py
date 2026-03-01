"""
Filter Paper Region Masking

Detects the circular filter paper region and masks out the background.
This ensures YOLO only detects microplastics on the filter, not the table.

Multi-strategy detection (ported from train_yolo.py predict):
  0. Full-coverage check  – if filter fills the whole image, skip circle fitting
  1. Edge-based Hough     – bilateral filter → Canny → HoughCircles (multiple param sets)
  2. Background invert    – HSV dark/wood mask → invert → largest contour → minEnclosingCircle
  3. Otsu contour         – Otsu threshold → largest contour → minEnclosingCircle
  4. Default fallback     – centred circle at 42 % of min(h,w)

All strategies run on a down-scaled image (≤ 1024 px longest side) for speed,
then results are mapped back to the original resolution.
"""

import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_MAX_DIM = 1024  # longest side for detection – keeps runtime < 0.5 s


def _downscale(img):
    """Return (small_img, scale_factor).  scale_factor maps small → original."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= _MAX_DIM:
        return img, 1.0
    scale = longest / _MAX_DIM
    new_w, new_h = int(w / scale), int(h / scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


# ---------------------------------------------------------------------------
# Strategy 0: full-coverage check
# ---------------------------------------------------------------------------

def _check_full_coverage(gray):
    """Return True if the filter paper fills the entire image (no dark bg)."""
    h, w = gray.shape[:2]
    dark_pixels = np.sum(gray < 100)
    dark_ratio = dark_pixels / (h * w)

    edge_margin = min(20, h // 20, w // 20)
    if edge_margin < 2:
        edge_margin = 2
    top    = gray[:edge_margin, :].mean()
    bottom = gray[-edge_margin:, :].mean()
    left   = gray[:, :edge_margin].mean()
    right  = gray[:, -edge_margin:].mean()
    edge_brightness = np.mean([top, bottom, left, right])

    return dark_ratio < 0.05 and edge_brightness > 150


# ---------------------------------------------------------------------------
# Strategy 1: Edge-based Hough Circle Detection
# ---------------------------------------------------------------------------

def _detect_edge_hough(gray, height, width):
    """Multi-param Hough on bilateral-filtered + Canny edges."""
    min_dim = min(height, width)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    param_sets = [
        {'dp': 1.2, 'param1': 50,  'param2': 30, 'minR': int(min_dim * 0.30), 'maxR': int(min_dim * 0.50)},
        {'dp': 1.5, 'param1': 80,  'param2': 40, 'minR': int(min_dim * 0.25), 'maxR': int(min_dim * 0.55)},
        {'dp': 1.0, 'param1': 100, 'param2': 25, 'minR': int(min_dim * 0.35), 'maxR': int(min_dim * 0.48)},
        {'dp': 2.0, 'param1': 50,  'param2': 50, 'minR': int(min_dim * 0.30), 'maxR': int(min_dim * 0.52)},
    ]

    img_cx, img_cy = width // 2, height // 2
    best_circle = None

    for p in param_sets:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=p['dp'], minDist=min_dim // 2,
            param1=p['param1'], param2=p['param2'],
            minRadius=p['minR'], maxRadius=p['maxR'],
        )
        if circles is None:
            continue
        circles = np.round(circles[0, :]).astype(int)
        for c in circles:
            dist = np.sqrt((c[0] - img_cx) ** 2 + (c[1] - img_cy) ** 2)
            if dist < min_dim * 0.3:
                if best_circle is None or c[2] > best_circle[2]:
                    best_circle = c
        if best_circle is not None:
            break

    if best_circle is not None:
        return int(best_circle[0]), int(best_circle[1]), int(best_circle[2]), "edge-Hough"
    return None


# ---------------------------------------------------------------------------
# Strategy 2: Background invert (HSV dark/wood detection)
# ---------------------------------------------------------------------------

def _detect_background_invert(gray, hsv, height, width):
    """Detect dark/wood background, invert to get filter paper, fit circle."""
    min_dim = min(height, width)

    # Dark background
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 120])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    # Wood-coloured background
    lower_wood = np.array([5, 30, 20])
    upper_wood = np.array([25, 200, 150])
    wood_mask = cv2.inRange(hsv, lower_wood, upper_wood)

    bg_mask = cv2.bitwise_or(dark_mask, wood_mask)
    filter_mask = cv2.bitwise_not(bg_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    filter_mask = cv2.morphologyEx(filter_mask, cv2.MORPH_CLOSE, kernel)
    filter_mask = cv2.morphologyEx(filter_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(filter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest)

    if contour_area < height * width * 0.10:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    cx, cy, radius = int(cx), int(cy), int(radius)

    circle_area = np.pi * radius * radius
    circularity = contour_area / circle_area if circle_area > 0 else 0

    if circularity > 0.6 and radius > min_dim * 0.20:
        return cx, cy, radius, "background-invert"
    return None


# ---------------------------------------------------------------------------
# Strategy 3: Otsu contour (from previous version)
# ---------------------------------------------------------------------------

def _detect_otsu_contour(gray, height, width):
    """Otsu threshold → largest contour → minEnclosingCircle."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.05 * height * width:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return int(round(cx)), int(round(cy)), int(round(radius)), "otsu-contour"


def detect_filter_circle(image_path: str, debug: bool = False):
    """
    Detect the circular filter paper in the image.
    
    Args:
        image_path: Path to the image
        debug: If True, save debug visualization
        
    Returns:
        center: (x, y) center of the filter (None if full-coverage)
        radius: radius of the filter in pixels (None if full-coverage)
        mask: binary mask of the filter region
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    center, radius, mask, method, full_coverage = detect_filter_circle_from_array(img)
    
    if debug and not full_coverage:
        debug_img = img.copy()
        cv2.circle(debug_img, center, radius, (0, 255, 0), 3)
        cv2.circle(debug_img, center, 5, (0, 0, 255), -1)
        debug_path = Path(image_path).parent / f"{Path(image_path).stem}_filter_debug.png"
        cv2.imwrite(str(debug_path), debug_img)
        print(f"Debug image saved: {debug_path}")
    
    return center, radius, mask


def detect_filter_circle_from_array(img: np.ndarray):
    """
    Detect the circular filter paper from a BGR numpy array.

    Runs a cascade of strategies on a down-scaled copy for speed:
      0. Full-coverage check (skip if filter fills entire image)
      1. Edge-based Hough (multiple parameter combos)
      2. Background invert (HSV dark/wood → invert → fit circle)
      3. Otsu contour (threshold → largest contour → fit circle)
      4. Default fallback (centred, 42 % of min dimension)

    Returns:
        center: (x, y) centre of the filter  — *None if full_coverage*
        radius: radius in pixels             — *None if full_coverage*
        mask: binary mask (255 inside filter) — full image of 255s if full_coverage
        method: str describing which strategy succeeded
        full_coverage: bool — True when filter fills the whole image
    """
    height, width = img.shape[:2]

    # ---- Downscale for speed ----
    small, scale = _downscale(img)
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # ---- Strategy 0: full-coverage check ----
    if _check_full_coverage(gray):
        mask = np.full((height, width), 255, dtype=np.uint8)
        print(f"Filter detection: full-coverage (no circle needed)")
        return None, None, mask, "full-coverage", True

    # ---- Strategy 1: Edge-based Hough ----
    result = _detect_edge_hough(gray, sh, sw)

    # ---- Strategy 2: Background invert ----
    if result is None:
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        result = _detect_background_invert(gray, hsv, sh, sw)

    # ---- Strategy 3: Otsu contour ----
    if result is None:
        result = _detect_otsu_contour(gray, sh, sw)

    # ---- Scale back & build mask ----
    if result is not None:
        sx, sy, sr, method = result
        center = (int(round(sx * scale)), int(round(sy * scale)))
        radius = int(round(sr * scale))
        # Shrink 2 % inward to cut edge artefacts (tape, shadow)
        radius = int(radius * 0.98)
        print(f"Detected filter: center={center}, radius={radius}, method={method}")
    else:
        # Strategy 4: default fallback
        center = (width // 2, height // 2)
        radius = int(min(height, width) * 0.42)
        method = "default-center"
        print(f"Warning: filter circle fallback: center={center}, radius={radius}")

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    return center, radius, mask, method, False


def mask_background(image_path: str, output_path: str = None):
    """
    Mask out the background, keeping only the filter paper region.
    
    Args:
        image_path: Path to input image
        output_path: Path to save masked image (default: adds _masked suffix)
        
    Returns:
        output_path: Path to the masked image
    """
    # Detect filter region
    center, radius, mask = detect_filter_circle(image_path, debug=True)
    
    # Read original image
    img = cv2.imread(image_path)
    
    # Apply mask - set background to black
    masked = cv2.bitwise_and(img, img, mask=mask)
    
    # Or set background to white (may work better for some models)
    # background = np.ones_like(img) * 255
    # masked = np.where(mask[:,:,np.newaxis] == 255, img, background)
    
    # Save masked image
    if output_path is None:
        output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_masked.png")
    
    cv2.imwrite(output_path, masked)
    print(f"Masked image saved: {output_path}")
    
    return output_path


def filter_detections_by_region(detections, center, radius, image_shape):
    """
    Filter YOLO detections to only include those within the filter region.
    
    Args:
        detections: YOLO detection boxes (x1, y1, x2, y2 format)
        center: (x, y) center of filter
        radius: radius of filter
        image_shape: (height, width) of image
        
    Returns:
        filtered_indices: indices of detections within the filter
    """
    filtered = []
    height, width = image_shape[:2]
    
    for i, box in enumerate(detections):
        # Get center of detection box
        x1, y1, x2, y2 = box[:4]
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        # Check if detection center is within filter circle
        dist = np.sqrt((box_center_x - center[0])**2 + (box_center_y - center[1])**2)
        
        if dist <= radius:
            filtered.append(i)
    
    return filtered


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mask filter paper region')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, help='Output path for masked image')
    
    args = parser.parse_args()
    
    mask_background(args.image, args.output)
