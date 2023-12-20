import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from white_image_removal import remove_whiteimages

# Set the root directory
root_directory = 'images/buoys'

# remove white images
remove_whiteimages()


def find_mostly_black_images(directory, black_threshold=10):
    black_images = []
    # Search in each subdirectory of the given directory
    for sub_dir in glob.glob(f'{directory}/*'):
        images = glob.glob(f'{sub_dir}/*.jpg')
        for image_path in tqdm(images, desc=f"Checking black images in {sub_dir}"):
            img = cv2.imread(image_path)
            if np.mean(img) < black_threshold:
                black_images.append(image_path)
    return black_images

def remove_mostly_black_images(root_directory, black_threshold=10):
    for sub_dir in glob.glob(f'{root_directory}/*'):
        black_images = find_mostly_black_images(sub_dir, black_threshold)
        for image_path in tqdm(black_images, desc="Removing black images"):
            os.remove(image_path)
            print(f"Removed black image: {image_path}")
