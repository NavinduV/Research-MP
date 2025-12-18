import cv2
import os

def load_images_from_folder(folder_path):
    """
    Loads all images from a folder and returns them as a list.
    """
    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder_path, file))
            if img is not None:
                images.append(img)
    return images


def stitch_images(image_list):
    """
    Uses OpenCV's stitcher to combine overlapping images.
    """
    stitcher = cv2.Stitcher_create(cv2.STITCHER_PANORAMA)
    status, stitched = stitcher.stitch(image_list)

    if status != cv2.Stitcher_OK:
        raise RuntimeError("Image stitching failed")

    return stitched


def stitch_folder(input_folder, output_path):
    """
    Stitches all images inside a folder and saves the output.
    """
    images = load_images_from_folder(input_folder)
    stitched_image = stitch_images(images)
    cv2.imwrite(output_path, stitched_image)


if __name__ == "__main__":
    stitch_folder(
        input_folder="data/raw/sample_tiles",
        output_path="data/stitched/stitched_filter.png"
    )
