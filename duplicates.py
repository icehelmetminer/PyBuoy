import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_image(image_path):
    """Load an image and cut the bottom 30 pixels."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    # return img[:-30, :, :]  # Remove the bottom 30 pixels
    return img #todo: testing without removing bottom 30 pixels

def are_images_similar(img1, img2, deviation_threshold=0.005):
    """Check if two images are similar within a certain deviation threshold."""
    try:
        difference = cv2.absdiff(img1, img2)
        gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        non_zero_count = np.count_nonzero(gray_difference)
        total_pixels = img1.shape[0] * img1.shape[1]
        diff_percentage = non_zero_count / total_pixels
        return diff_percentage < deviation_threshold
    except Exception as e:
        print(e)
        return False

def remove_similar_images(directory):
    """Remove similar images within each buoy sub-directory."""
    for sub_dir in tqdm(glob.glob(f'{directory}/*'), desc=f"Processing buoy sub-directories"):
        images = glob.glob(f'{sub_dir}/*.jpg')
        for i, img_path1 in tqdm(enumerate(images),
                                 desc=f"Checking images in {sub_dir}",
                                 leave=False):
            if 'panels' in img_path1:
                continue
            img1 = preprocess_image(img_path1)
            if img1 is None:
                continue
            for img_path2 in images[i+1:]:
                img2 = preprocess_image(img_path2)
                if img2 is not None and are_images_similar(img1, img2):
                    os.remove(img_path2)
                    print(f"Removed similar image: {img_path2}")

if __name__ == "__main__":
    root_directory = 'images/buoys'
    print('Removing similar images')
    remove_similar_images(root_directory)
