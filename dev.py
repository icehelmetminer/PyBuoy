import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import random

REDO = True

def find_extreme_images(directory, low_threshold=5, high_threshold=250):
    extreme_images = {'white': [], 'black': []}
    print(f"Checking images in {len(glob.glob(f'{directory}/*'))} folders")
    for sub_dir in glob.glob(f'{directory}/*'):
        if 'panels' in sub_dir:
            continue  # Skip already vetted 'panels' subdirectory
        images = glob.glob(f'{sub_dir}/*.jpg')
        for image_path in tqdm(images, desc=f"Checking images in {sub_dir}"):
            img = cv2.imread(image_path)
            if img is None:
                continue  # Skip if the image wasn't loaded properly
            if np.mean(img) > high_threshold:
                extreme_images['white'].append(image_path)
            # elif is_mostly_black(img, low_threshold):
            #     extreme_images['black'].append(image_path)
        print(f"Found {len(extreme_images['white'])} white images and {len(extreme_images['black'])} black images")
    return extreme_images

def is_mostly_black(img, low_threshold):
    """Check if the image is mostly black, ignoring other colors."""
    result = True
    #^ Check 1 - Other Colors
    # Check if there are any significant areas that are not black
    if np.any(img > low_threshold):
        # Check each channel separately to see if it has any significant non-black areas
        for channel in range(3):  # For B, G, R channels
            if np.mean(img[:,:,channel]) > low_threshold:
                result = False
    #^ Check 2 - Black only
    # Check if the majority of the image is black
    if np.mean(img) >= low_threshold:
        result = False

    #^ Determine how many pixels are black
    # Check if the majority of the pixels are black
    if np.mean(img) >= low_threshold:
        result = False
    return result


def remove_images(image_paths):
    for image_path in tqdm(image_paths, desc="Removing images"):
        os.remove(image_path)
        print(f"Removed image: {image_path}")

# Example usage:
root_directory = 'images/buoys'

# if the duplicate csv files don't exist, then search for the extreme images and duplicates and save them as csv files
# if the duplicate csv files do exist, then read them in
#& Extreme Images
if not os.path.exists('extreme_images.csv') or REDO:
    extreme_images = find_extreme_images(root_directory)
    # save the file paths for all extreme images as a csv
else:
    extreme_images = {'white': [], 'black': []}
    with open('extreme_images.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            extreme_images['white'].append(row[0])
            extreme_images['black'].append(row[1])


random.seed(42)


print(f"Found {len(extreme_images['white'])} white images and {len(extreme_images['black'])} black images")



remove_images(extreme_images['white'])  # Remove all white images
# remove_images(extreme_images['black'])  # Remove all black images
