"""
Filter Paper Region Masking

Detects the circular filter paper region and masks out the background.
This ensures YOLO only detects microplastics on the filter, not the table.
"""

import cv2
import numpy as np
from pathlib import Path


def detect_filter_circle(image_path: str, debug: bool = False):
    """
    Detect the circular filter paper in the image.
    
    Args:
        image_path: Path to the image
        debug: If True, save debug visualization
        
    Returns:
        center: (x, y) center of the filter
        radius: radius of the filter in pixels
        mask: binary mask of the filter region
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    height, width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(height, width) // 2,  # Only detect one main circle
        param1=50,
        param2=30,
        minRadius=min(height, width) // 4,  # Filter is at least 1/4 of image
        maxRadius=min(height, width) // 2   # Filter is at most 1/2 of image
    )
    
    if circles is None:
        # Fallback: assume filter is centered and takes most of the image
        center = (width // 2, height // 2)
        radius = int(min(height, width) * 0.4)
        print(f"Warning: Could not detect filter circle. Using default: center={center}, radius={radius}")
    else:
        # Get the largest/most confident circle
        circles = np.round(circles[0, :]).astype("int")
        # Sort by radius (largest first)
        circles = sorted(circles, key=lambda c: c[2], reverse=True)
        x, y, r = circles[0]
        center = (x, y)
        radius = r
        print(f"Detected filter: center={center}, radius={radius}")
    
    # Create circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    if debug:
        # Save debug visualization
        debug_img = img.copy()
        cv2.circle(debug_img, center, radius, (0, 255, 0), 3)
        cv2.circle(debug_img, center, 5, (0, 0, 255), -1)
        debug_path = Path(image_path).parent / f"{Path(image_path).stem}_filter_debug.png"
        cv2.imwrite(str(debug_path), debug_img)
        print(f"Debug image saved: {debug_path}")
    
    return center, radius, mask


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
