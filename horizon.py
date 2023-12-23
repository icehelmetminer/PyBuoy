# import cv2
# import numpy as np
# import os

# def cut_into_panels(image):
#     """
#     Cut a raw image into 6 panels.
#     :param image: The rotated image.
#     :return: A list of images, each containing a single panel.
#     """
#     # Get the dimensions of the image
#     height, width = image.shape[:2]

#     # Calculate the height of each panel
#     panel_height = height // 6

#     # Cut the image into 6 panels
#     panels = []
#     for i in range(6):
#         start = i * panel_height
#         end = start + panel_height
#         panels.append(image[start:end, :])

#     return panels

# def process_image(image_path):

#     # Load the image to cut into panels

#     image = cv2.imread(image_path)

#     # cut the image into panels
#     panels = cut_into_panels(image)

#     # Save the panels in the panels directory
#     for i, panel in enumerate(panels):
#         panel_path = os.path.join('modified_panels', f'{i}.jpg')
#         cv2.imwrite(panel_path, panel)

#     for panel_image in os.listdir('modified_panels'):
#         panel_image_path = os.path.join('modified_panels', panel_image)
#         process_panel_image(panel_image_path)

#         # Crop out the bottom banner
#         cropped_image = image[:-37, :]  # Assuming the banner is 37 pixels high

#         # Convert to grayscale and detect edges
#         gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150)

#         # Find horizon and rotate
#         rotation_angle = find_horizon_line(edges)
#         rotated_image = rotate_image(cropped_image, rotation_angle)

#         # Save the modified image in the modified_panels directory
#         modified_image_path = os.path.join('modified_panels', os.path.basename(image_path))
#         cv2.imwrite(modified_image_path, rotated_image)

#     return modified_image_path


# # Example usage
# # test on this image: images/buoys/panels/41002_20231218151103.jpg
# modified_image_path = process_image('images/buoys/panels/41002_20231218151103.jpg')

# print(f"Modified image saved: {modified_image_path}")


#* ----------------

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def cut_into_panels(image):
    """
    Cut a raw image into 6 panels horizontally.
    :param image: The rotated image.
    :return: A list of images, each containing a single panel.
    """
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the width of each panel
    panel_width = width // 6

    # Cut the image into 6 panels
    panels = [image[:, i * panel_width:(i + 1) * panel_width] for i in range(6)]

    return panels


def find_horizon_line(edge_img):
    """
    Find the most prominent horizon line in an edge-detected image and return the angle to rotate the image.
    The horizon line is assumed to be the longest line near the horizontal orientation.

    :param edge_img: Edge-detected input image.
    :return: The angle in degrees to rotate the original image.
    """
    # Detect lines in the image using the Hough Transform
    lines = cv2.HoughLines(edge_img, 1, np.pi / 180, 150)
    if lines is not None:
        # Initialize the maximum length and corresponding angle
        max_len = 0
        angle_to_rotate = 0

        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Calculate the line length
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Check if this line is the longest near-horizontal line found so far
            if line_length > max_len and abs(theta - np.pi / 2) < np.pi / 36:  # 5 degrees tolerance for being horizontal
                max_len = line_length
                angle_to_rotate = (theta - np.pi / 2) if theta > np.pi / 2 else (theta - np.pi / 2 + np.pi)
                angle_to_rotate = np.rad2deg(angle_to_rotate)

        return -angle_to_rotate  # Negative to compensate for image coordinate system
    else:
        # If no lines are found, return 0 as no rotation is needed
        return 0


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    print(f'Rotation Data: \n\tangle = {angle}\n\tcenter = {center}\n\tM = {M}')
    return rotated


def adaptive_cropping(image):
    """
    Crop an image by finding the row that is closest to the bottom of the image that is NOT black and crop the image from there.

    # Example usage:
    # Load the image
    image_path = 'path_to_your_image.jpg'
    image = cv2.imread(image_path)

    # Crop the image
    cropped_image = adaptive_cropping(image)

    # Save the cropped image or proceed with other operations
    cv2.imwrite('path_to_cropped_image.jpg', cropped_image)


    :param image: The image to crop.
    :return: The cropped image.

    """
    # Convert the image to grayscale to simplify the processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the first row from the bottom that is not all black (0)
    for i in range(gray_image.shape[0] - 1, -1, -1):
        if not np.all(gray_image[i] == 0):
            cropped_image = image[:i, :]
            return cropped_image

    # If no non-black pixel is found, return the original image
    return image





def process_panel_image(panel, panel_index, output_dir='modified_panels'):
    """
    Process a single panel image by cropping, rotating, and saving it.
    :param panel: The panel image to process.
    :param panel_index: The index of the panel for naming the saved file.
    :param output_dir: The directory where the processed panel will be saved.
    """
    # Crop out the bottom banner
    # cropped_panel = panel[:-37, :]  # Assuming the banner is 37 pixels high
    #note: make this adaptive. Find the pixel that is closest to the bottom of the image that is NOT black and crop the image from there
    # use the adaptive_cropping function to find the pixel that is closest to the bottom of the image that is NOT black and crop the image from there. Check the syntax within the docstring of the function (above) for how to use it.
    cropped_panel = adaptive_cropping(panel)

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(cropped_panel, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find horizon and rotate
    rotation_angle = find_horizon_line(edges)
    rotated_panel = rotate_image(cropped_panel, rotation_angle)

    # Save the modified panel
    os.makedirs(output_dir, exist_ok=True)
    modified_image_path = os.path.join(output_dir, f'panel_{panel_index}_modified.jpg')
    cv2.imwrite(modified_image_path, rotated_panel)
    print(f"Modified panel saved: {modified_image_path}")

def process_image(image_path):
    # Load the image to cut into panels
    image = cv2.imread(image_path)

    # Cut the image into panels
    panels = cut_into_panels(image)

    # Process each panel and save it
    for i, panel in enumerate(panels):
        process_panel_image(panel, i)
        print(f"Processed panel {i}")


# Call process_image with the path to your image
process_image('images/buoys/panels/41002_20231218151103.jpg')