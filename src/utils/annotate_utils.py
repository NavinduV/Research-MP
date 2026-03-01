import numpy as np
import cv2

def mask_to_bbox(mask):
    """
    Converts a binary mask to bounding box [x, y, w, h].
    """
    ys, xs = np.where(mask > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def crop_instance(image, mask):
    """
    Crops an object from image using its mask.
    """
    bbox = mask_to_bbox(mask)
    x, y, w, h = bbox
    cropped = image[y:y+h, x:x+w]
    return cropped
