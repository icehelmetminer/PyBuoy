import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_image(image_path):

    """
    The preprocess_image function takes an image path as input and returns a preprocessed image.
    The preprocessing steps are:
        1. Read the image from disk using cv2.imread()
        2. If the returned value is None, return None (this means that there was an error reading the file)
        3. Remove 30 pixels from the bottom of each frame to remove any car hoods or other artifacts that may be present in some frames.

    :param image_path: Pass the path of the image to be preprocessed
    :return: The image without the bottom 30 pixels
    :doc-author: Trelent
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    # return img[:-30, :, :]  # Remove the bottom 30 pixels
    return img #todo: testing without removing bottom 30 pixels

def are_images_similar(img1, img2, deviation_threshold=0.005, standard_size=(100, 100)):

    """
    The are_images_similar function takes two images and compares them to see if they are similar.
    It does this by resizing the images to a standard size, then comparing the difference between each pixel in both images.
    If there is a deviation of more than 0.005% (default) between any pixels, it returns False.

    :param img1: Compare the image to img2
    :param img2: Compare the image with img2
    :param deviation_threshold: Determine the percentage of pixels that are allowed to be different between two images
    :param standard_size: Resize the images to a standard size before comparing them
    :param 100): Set the size of the image
    :return: True if the two images are similar, false otherwise
    :doc-author: Trelent
    """
    img1 = cv2.resize(img1, standard_size)
    img2 = cv2.resize(img2, standard_size)

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

    """
    The remove_similar_images function takes a directory as an argument and removes all images in that directory
    that are similar to other images in the same sub-directory. The function first iterates through each sub-directory
    in the given directory, then iterates through each image within that sub-directory. For every image, it compares
    the current image with every other image in the same sub-directory (excluding itself). If two images are found to be
    similar, one of them is removed from disk.

    :param directory: Specify the directory where the images are located
    :return: A list of similar images
    :doc-author: Trelent
    """
    total_len = len(glob.glob(f'{directory}/*'))
    for sub_dir in tqdm(glob.glob(f'{directory}/*'), desc=f"Processing buoy sub-directories", total = total_len):
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
