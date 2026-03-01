import cv2
import numpy as np

def live_preprocessing_demo(image_path):
    """Live preprocessing demo with adjustable parameters."""
    def nothing(x):
        pass

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Create a window
    cv2.namedWindow('Live Preprocessing')

    # Create trackbars for adjusting parameters
    cv2.createTrackbar('Blur', 'Live Preprocessing', 1, 50, nothing)
    cv2.createTrackbar('CLAHE Clip', 'Live Preprocessing', 2, 10, nothing)
    cv2.createTrackbar('Brightness', 'Live Preprocessing', 50, 100, nothing)
    cv2.createTrackbar('Contrast', 'Live Preprocessing', 50, 100, nothing)

    while True:
        # Get current positions of trackbars
        blur_ksize = cv2.getTrackbarPos('Blur', 'Live Preprocessing')
        clahe_clip = cv2.getTrackbarPos('CLAHE Clip', 'Live Preprocessing')
        brightness = cv2.getTrackbarPos('Brightness', 'Live Preprocessing')
        contrast = cv2.getTrackbarPos('Contrast', 'Live Preprocessing')

        # Ensure blur kernel size is odd
        if blur_ksize % 2 == 0:
            blur_ksize += 1

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

        # Convert to LAB and apply CLAHE
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        # Adjust brightness and contrast
        adjusted = cv2.convertScaleAbs(enhanced, alpha=contrast / 50, beta=brightness - 50)

        # Display the result
        cv2.imshow('Live Preprocessing', adjusted)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "data/raw/sample.jpg"  # Replace with your image path
    live_preprocessing_demo(image_path)