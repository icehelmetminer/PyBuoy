import cv2
import numpy as np
import os
from datetime import datetime
from config import IMAGE_DIRECTORY, BUOYCAM_IDS, COLLAGE_DIRECTORY
from logger import setup_logger

logger = setup_logger()

def is_valid_image(image, threshold=10):
    return np.mean(image) >= threshold and np.mean(image) <= 245

def enhance_image(image):
    """
    Enhances the image by applying CLAHE and adjusting color saturation.

    Args:
        image (numpy.ndarray): The image to be enhanced.

    Returns:
        numpy.ndarray: The enhanced image.
    """
    try:
        # Convert to YUV color space
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Apply CLAHE to the Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])

        # Convert back to BGR color space
        enhanced_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

        # Adjust saturation in HSV color space
        image_hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
        image_hsv[:, :, 1] = cv2.multiply(image_hsv[:, :, 1], 1.1)  # Increase saturation by 10%
        enhanced_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    except Exception as e:
        logger.error(f"Failed to enhance image: {e}")
        return image  # Return original image if enhancement fails

    return enhanced_image

def process_images(image_content, buoycam_id):
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{buoycam_id}_{timestamp}.jpg"
    image_path = os.path.join(IMAGE_DIRECTORY, buoycam_id, filename)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(image_content)
    image = cv2.imread(image_path)
    if is_valid_image(image):
        enhanced_image = enhance_image(image)
        cv2.imwrite(image_path, enhanced_image)
    else:
        logger.warning(f"buoy {buoycam_id}: invalid image")
