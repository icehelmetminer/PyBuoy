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
    """
    The find_extreme_images function takes a directory as an argument and returns a dictionary of lists.
    The keys are 'white' and 'black', the values are lists of paths to images that have been determined to be
    either too white or too black, respectively. The function will recursively search through all subdirectories
    of the given directory for .jpg files, load them into memory using OpenCV's imread() function, calculate their mean pixel value (a number between 0-255), and compare it against two thresholds: low_threshold (default 5) and high_threshold (default 250). If the image's mean

    :param directory: Specify the directory where all of the images are stored
    :param low_threshold: Determine if an image is mostly black
    :param high_threshold: Determine if an image is mostly white
    :return: A dictionary of lists
    :doc-author: Trelent
    """

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
        # print(f"Found {len(extreme_images['white'])} white images and {len(extreme_images['black'])} black images")
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
    """
    The remove_images function takes in a list of image paths and removes them from the file system.

    :param image_paths: Pass in the list of image paths that we want to remove
    :return: A list of the images that were removed
    :doc-author: Trelent
    """

    for image_path in tqdm(image_paths, desc="Removing images"):
        os.remove(image_path)
        print(f"Removed image: {image_path}")


def remove_whiteimages(root_directory='images/buoys'):
    """
    The remove_whiteimages function removes all images that are completely white or black from the dataset.
        It does this by first finding all of the extreme images, and then removing them from their respective directories.

    :return: The number of white images removed
    :doc-author: Trelent
    """

    # Example usage:
    # root_directory = 'images/buoys'

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


    # print(f"Found {len(extreme_images['white'])} white images and {len(extreme_images['black'])} black images")

    remove_images(extreme_images['white'])  # Remove all white images
    # remove_images(extreme_images['black'])  # Remove all black images
