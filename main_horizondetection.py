import argparse
import os

import cv2
import matplotlib.pyplot as plt

# from horizon_utils import detect_horizon_line
# credit to https://github.com/sallamander/horizon-detection/blob/master/utils.py


import cv2

import numpy as np
# credit to  https://github.com/sallamander/horizon-detection/blob/master/utils.py

def detect_horizon_line(image_grayscaled):
    """Detect the horizon's starting and ending points in the given image

    The horizon line is detected by applying Otsu's threshold method to
    separate the sky from the remainder of the image.

    :param image_grayscaled: grayscaled image to detect the horizon on, of
     shape (height, width)
    :type image_grayscale: np.ndarray of dtype uint8
    :return: the (x1, x2, y1, y2) coordinates for the starting and ending
     points of the detected horizon line
    :rtype: tuple(int)
    """

    msg = ('`image_grayscaled` should be a grayscale, 2-dimensional image '
           'of shape (height, width).')
    assert image_grayscaled.ndim == 2, msg
    image_blurred = cv2.GaussianBlur(image_grayscaled, ksize=(3, 3), sigmaX=0)

    _, image_thresholded = cv2.threshold(
        image_blurred, thresh=0, maxval=1,
        type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )
    image_thresholded = image_thresholded - 1
    image_closed = cv2.morphologyEx(image_thresholded, cv2.MORPH_CLOSE,
                                    kernel=np.ones((9, 9), np.uint8))

    horizon_x1 = 0
    horizon_x2 = image_grayscaled.shape[1] - 1
    horizon_y1 = max(np.where(image_closed[:, horizon_x1] == 0)[0])
    horizon_y2 = max(np.where(image_closed[:, horizon_x2] == 0)[0])

    return horizon_x1, horizon_x2, horizon_y1, horizon_y2

# def parse_args():
#     """Parse command line arguments"""

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         '--dirpath_input_images', type=str, required=True,
#         help='Absolute directory path to images to detect the horizon on.'
#     )
#     parser.add_argument(
#         '--dirpath_output_images', type=str, required=True,
#         help='Absolute directory path to save output images in.'
#     )

#     args = parser.parse_args()
#     return args

def main():
    """Main logic"""

    args = parse_args()

    dirpath_input_images = args.dirpath_input_images
    dirpath_output_images = args.dirpath_output_images
    msg = ('`dirpath_input_images` and `dirpath_output_images` cannot point to'
           'the same directory.')
    assert dirpath_input_images != dirpath_output_images, msg
    os.mkdir(dirpath_output_images)

    for fname_image in os.listdir(dirpath_input_images):
        fpath_image = os.path.join(dirpath_input_images, fname_image)

        fig, axes = plt.subplots(1, 2)
        image = cv2.imread(fpath_image)
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        horizon_x1, horizon_x2, horizon_y1, horizon_y2 = detect_horizon_line(
            image_grayscale
        )
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Original Image')

        axes[1].imshow(image_grayscale, cmap='gray')
        axes[1].axis('off')
        axes[1].plot(
            (horizon_x1, horizon_x2), (horizon_y1, horizon_y2),
            color='r', linewidth='2'
        )
        axes[1].set_title('Grayscaled Image\n with Horizon Line (Red)')

        fpath_save = os.path.join(dirpath_output_images, fname_image)
        fpath_save = fpath_save.replace('jpg', 'png')
        fig.savefig(fpath_save, bbox_inches='tight')
        plt.close()

# import cv2

# from utils import detect_horizon_line

# fpath_image = '/absolute/path/to/image'
# image = cv2.imread(fpath_image)
# image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# horizon_x1, horizon_x2, horizon_y1, horizon_y2 = (
#     detect_horizon_line(image_grayscale)
# )

if __name__ == '__main__':
    main()