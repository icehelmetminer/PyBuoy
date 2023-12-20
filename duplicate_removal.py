import csv
import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
    #? Interesting Error Codes from the duplicate_removal.py file
    - OpenCV(4.8.1) /Users/runner/work/opencv-python/opencv-python/opencv/modules/core/src/arithm.cpp:650: error: (-209:Sizes of input arguments do not match) The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array' in function 'arithm_op'


    Notes:
    This uses a default sample of 50 pixels to check for similarity between images.
    - If the images are similar, then they are considered duplicates.
    - If the images are not similar, then they are not considered duplicates.
"""


#^ Global Variables
REDO = True
VISUAL_CHECK = False
MASTER_DUPLICATE_PATH_LIST = []
# load the master duplicate list if it exists from a csv file
try:
    # use pandas
    df = pd.read_csv('duplicate_images.csv')
    MASTER_DUPLICATE_PATH_LIST = df.values.tolist()
except FileNotFoundError:
    # create the csv file with pandas
    df = pd.DataFrame(MASTER_DUPLICATE_PATH_LIST)
    df.to_csv('duplicate_images.csv', index=False, header=['image1', 'image2'])
except Exception as e:
    print(e)

def get_image_hash(img, num_samples=50):
    """
    The get_image_hash function takes an image and returns a hash of the image.

    :param img: Pass the image to be hashed
    :param num_samples: Determine the number of pixels to sample from the image
    :return: A hash of the image
    :doc-author: Trelent
    """

    try:
        random_pixels = (img[random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)] for _ in range(num_samples))
        return hash(tuple(pixel.tobytes() for pixel in random_pixels))
    except Exception as e:
        print(e)
        return None


def are_images_similar(img1, img2, deviation_threshold=0.005):
    """
    The are_images_similar function takes two images as input and returns a boolean value indicating whether the images are similar.
    The function calculates the percentage of different pixels between the two images, and if that percentage is less than a deviation threshold,
    the function returns True. Otherwise it returns False.

    :param img1: Pass the first image to compare
    :param img2: Compare the image to a second image
    :param deviation_threshold: Determine how similar two images are
    :return: A boolean value
    :doc-author: Trelent
    """

    try:
        # Calculate the absolute difference image
        difference = cv2.absdiff(img1, img2)
        # Convert difference to grayscale
        gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        # Count non-zero pixels within the difference image
        non_zero_count = np.count_nonzero(gray_difference)
        # Calculate the percentage of different pixels
        total_pixels = img1.shape[0] * img1.shape[1]
        diff_percentage = non_zero_count / total_pixels

        # If the percentage of different pixels is less than the deviation threshold, images are similar
        return diff_percentage < deviation_threshold
    except Exception as e:
        print(e)
        return False

def find_duplicates(directory):
    """
    The find_duplicates function takes a directory as input and returns a list of lists.
    Each sublist contains two elements: the first element is the path to an image, and
    the second element is the path to another image that has been deemed similar by our
    are_images_similar function. The find_duplicates function should recursively search all
    subdirectories in the given directory for images, compare each image with every other
    image in its subdirectory (but not itself), and add any pairs of similar images to its return value.

    :param directory: Specify the directory to search for duplicates in
    :return: A list of lists
    :doc-author: Trelent
    """

    duplicates = []
    total_duplicates = 0
    for sub_dir in tqdm(glob.glob(f'{directory}/*'), desc=f"Finding Duplicates", leave=False):
        if 'panels' in sub_dir:
            continue
        images = glob.glob(f'{sub_dir}/*.jpg')
        seen = []

        for image_path in tqdm(images,
                               desc=f"Checking images in {sub_dir}",
                               leave=False):
            img = cv2.imread(image_path)
            if img is None:
                continue

            # Compare the current image with all seen images in the same subdirectory
            is_duplicate = False
            for seen_img_path in seen:
                seen_img = cv2.imread(seen_img_path)
                if are_images_similar(img, seen_img):
                    duplicates.append([image_path, seen_img_path])  # Note the brackets to make it a list
                    is_duplicate = True
                    MASTER_DUPLICATE_PATH_LIST.append([image_path, seen_img_path])  # Append as list
                    total_duplicates += 1
                    break
            if not is_duplicate:
                seen.append(image_path)
        print(f'>> total duplicates found: {total_duplicates}')
        # save the master duplicate list and update the csv file
        # using pandas
        # Save the duplicates to a csv file using pandas
        df = pd.DataFrame(MASTER_DUPLICATE_PATH_LIST, columns=['image1', 'image2'])  # Specify column names
        df.to_csv('duplicate_images.csv', index=False)


    # Save the duplicates to a csv file
    with open('duplicate_images.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image1', 'image2'])
        writer.writerows(duplicates)

    return duplicates

def remove_duplicate_images():
    """
    The remove_duplicate_images function is used to remove duplicate images from a directory.

    :return: The list of duplicate images
    :doc-author: Trelent
    """
    # Example usage
    root_directory = 'images/buoys'

    # Check for duplicates and remove them
    if not os.path.exists('duplicate_images.csv') or REDO:
        duplicate_images = find_duplicates(root_directory)
    else:
        duplicate_images = []
        # read with pandas
        df = pd.read_csv('duplicate_images.csv')
        # two columns in the csv file ['image1', 'image2']
        for index, row in df.iterrows():
            duplicate_images.append([row['image1'], row['image2']])
        # we want to only keep the first image in the list for removal
        # we can do this by removing the second image in the list
        for image1, image2 in duplicate_images:
            if image2 in MASTER_DUPLICATE_PATH_LIST:
                MASTER_DUPLICATE_PATH_LIST.remove(image2)

    # Removing duplicates
    for image1, image2 in tqdm(duplicate_images, desc="Removing duplicate images"):
        # Before removal, we can add a visualization or confirmation step.
        # os.remove(image2)
        if VISUAL_CHECK:
            #^ Visualization and Confirmation of Duplicates
            img1 = cv2.imread(image1)
            img2 = cv2.imread(image2)
            # plot them on top of eachother one above the other to see if they are the same
            plt.subplot(1, 2, 1)
            plt.imshow(img1)
            plt.title('Image 1')
            plt.subplot(1, 2, 2)
            plt.imshow(img2)
            plt.title('Image 2')
            plt.show()
        print(f"Duplicate found: {image1} and {image2}")
        #^ Remove Duplicates
        # Remove the duplicate image
        os.remove(image2)
        print(f"Removed duplicate image: {image2}, \n\tkept {image1}")

    # Save the master duplicate list and update the csv file
    # using pandas
    # Save the duplicates to a csv file using pandas
    df = pd.DataFrame(MASTER_DUPLICATE_PATH_LIST, columns=['image1', 'image2'])  # Specify column names
    df.to_csv('duplicate_images.csv', index=False)

    return duplicate_images

def remove_duplicates_from_csv(csv_file):
    """
    Remove duplicate images based on entries in a CSV file.
    Assumes the CSV has columns 'image1' and 'image2'.
    """
    # Read the CSV file containing duplicates
    df = pd.read_csv(csv_file)

    # Iterate through each pair of duplicates
    for _, row in tqdm(df.iterrows(), desc="Processing duplicates", total=len(df)):
        # Remove the second image in each duplicate pair
        if os.path.exists(row['image2']):
            os.remove(row['image2'])
            print(f"Removed duplicate image: {row['image2']}")

if __name__ == "__main__":
    # Path to the CSV file with duplicates
    csv_file = 'duplicate_images.csv'
    print('Removing duplicate images')
    remove_duplicates_from_csv(csv_file)


# print('Removing duplicate images')
# remove_duplicate_images()