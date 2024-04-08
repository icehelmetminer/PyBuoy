import glob
import logging
import logging.handlers  # this is for the rotating file handler which means that the log file will not get too large and will be easier to read
import os
import time
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import numpy as np
import requests
from icecream import ic
from tqdm import tqdm
import random

from white_image_removal import remove_whiteimages
# from duplicate_removal import remove_duplicates, remove_duplicates_from_csv
from duplicates import remove_similar_images #* this works best, requires a root directory to be specified
# from dev.config import BUOYCAM_IDS
#note: remove_duplicates_from_csv relies on the csv file being in the same directory as the script, so it will need to be moved to the same directory as the script before running it.

## Author: Graham Waters
## Date: 12/17/2023
## Description: This script scrapes the NOAA buoycam images and creates a collage of the latest images from each buoycam. It also saves the latest images for each buoycam in a separate directory. The beauty of the worlds oceans is captured in these images, and this script allows you to view them all in one place.

#* Current Progress Note to Programmer ------
"""
    Using this file until the modules are working. main.py is not saving the images correctly and the image_processor.py is not working correctly. Until troubleshooting can happen, this file seems to work the best.

    Current Structure:
    ```md
    # PySeas
    * [.vscode/](./PySeas/.vscode)
    * [Pyseas_revived/](./PySeas/Pyseas_revived)
    * [__pycache__/](./PySeas/__pycache__)
    * [data/](./PySeas/data)
    * [docs/](./PySeas/docs)
    * [images/](./PySeas/images)
    * [legacy/](./PySeas/legacy)
    * [logs/](./PySeas/logs)
    * [models/](./PySeas/models)
    * [notebooks/](./PySeas/notebooks)
    * [panels/](./PySeas/panels)
    * [path/](./PySeas/path)
    * [sample/](./PySeas/sample)
    * [tests/](./PySeas/tests)
    * [.gitignore](./PySeas/.gitignore)
    * [LICENSE](./PySeas/LICENSE)
    * [pybuoy_final.py](./PySeas/pybuoy_final.py)

    #! The files that have been modularized are:
    * [config.py](./PySeas/config.py)
    * [image_processor.py](./PySeas/image_processor.py)
    * [logger.py](./PySeas/logger.py)
    * [main.py](./PySeas/main.py)
    * [scraper.py](./PySeas/scraper.py)
    ```

#note: the log file may get large quickly, implement a size checking parallel function to take out lines from the beginning of the file if it gets too large and keep it under 1 MB
#//: modified dev.py to only examine last 100 images if more than 100 in the folder.
Further thoughts: it would be nice to sort the panoramas by id before stitching them vertically so that they remain in the same order as the original images. This would make it easier to compare the panoramas to the original images. This could be done by sorting the list of images by id before stitching them together.

#todo items:
It appears some of the Buoy cameras don't turn off, they hold the latest image. So, instead of checking for black we need to be sure they have changed within the latest update period.

"""

#* -------------------------

#^ Set up Logging directory and file
#! if no log file is found, one will be created
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('logs/pybuoy.log'):
    open('/Users/grahamwaters/Library/Mobile Documents/com~apple~CloudDocs/PySeas/logs/.pybuoy.log.icloud', 'a').close()

# Initiate logging settings
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler('logs/pybuoy.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Globals
verbose = False # this is a boolean value that will be used to print out more information to the console
bypassing_hour = True #^ This is a boolean value that will be used to bypass the hour if it is not between the start and end hours
CLEANING_ACTIVE = True #^ This is a boolean value that will be used to determine if the cleaning function is active or not
IMAGES_DIR = '/Users/grahamwaters/Library/CloudStorage/GoogleDrive-graham.waters.business@gmail.com/My Drive/pyseas_images' #^ This is the directory where the images will be saved
# IMAGES_DIR = '/Volumes/Passport_2T/PySeas'
# Main execution
class BuoyCamScraper:
    def __init__(self, image_directory, buoycam_ids):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class with all of its attributes and methods.

        :param self: Represent the instance of the class
        :param image_directory: Set the directory where images will be saved
        :param buoycam_ids: Create a list of buoycam_ids
        :return: The image_directory and buoycam_ids
        :doc-author: Trelent
        """

        self.image_directory = image_directory
        self.buoycam_ids = buoycam_ids
        os.makedirs(self.image_directory, exist_ok=True)


    def is_image_white(self, image_path):
        """
        Determine if an image is completely white.

        :param image_path: Path to the image file
        :return: True if the image is completely white, False otherwise
        """
        with Image.open(image_path) as img:
            # Convert image to numpy array
            img_array = np.array(img)
            # Check if all pixels are white (you might adjust the threshold as necessary)
            return np.all(img_array > 250)

    def get_last_ten_image_paths(self, buoy_id):
        """
        Retrieve the file paths for the last ten images of a given buoy ID folder.

        :param buoy_id_folder_path: The folder path for the buoy ID.
        :return: A list of paths to the last ten images.
        """
        buoy_id_folder_path = f'/Users/grahamwaters/Library/CloudStorage/GoogleDrive-graham.waters.business@gmail.com/My Drive/pyseas_images/{buoy_id}'
        # List all files in the directory
        files = [os.path.join(buoy_id_folder_path, f) for f in os.listdir(buoy_id_folder_path) if os.path.isfile(os.path.join(buoy_id_folder_path, f))]

        # Sort the files by modification time in descending order
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Return the paths of the last ten images (or fewer if not enough images are available)
        return files[:10]

    def are_last_ten_images_white(self, buoycam_id):
        """
        Checks if the last ten images for a given buoy are completely white.

        :param buoycam_id: The ID of the buoy to check.
        :return: True if the last ten images are completely white, False otherwise.
        """
        # This is a placeholder list of image paths. You will need to replace this
        # with the actual logic to retrieve the last ten image paths for the buoy.
        last_ten_image_paths = self.get_last_ten_image_paths(buoycam_id)

        # Analyze each image
        for image_path in last_ten_image_paths:
            if not self.is_image_white(image_path):
                # If any image is not completely white, return False
                return False
        # If all images are white, return True
        return True

    def update_buoy_ids_file(self):
        """
        Updates the buoys_config.py file with the current list of buoycam_ids.
        """
        with open('buoys_config.py', 'w') as file:
            file.write("# buoys_config.py\nbuoy_ids = [\n")
            for buoycam_id in self.buoycam_ids:
                file.write(f'    "{buoycam_id}",\n')
            file.write("]\n")

    def scrape(self):
        """
        Modified scrape function that uses buoy IDs from a Python module.
        """
        from buoys_config import BUOYCAM_IDS as BUOYCAMS2


        # Assuming BUOYCAMS2 is another list of buoy IDs, possibly overlapping with PRIMARY_BUOY_IDS
        # BUOYCAMS2 = [...]  # This should be defined with your specific IDs

        # Convert PRIMARY_BUOY_IDS to a set for faster lookup
        primary_buoy_ids_set = set(PRIMARY_BUOY_IDS)

        # Select 10 unique random IDs from BUOYCAMS2 that are not already in the primary list
        selected_ids = set()
        while len(selected_ids) < 10:
            random_id = random.choice(BUOYCAMS2)
            if random_id not in primary_buoy_ids_set:
                selected_ids.add(random_id)

        # Convert selected_ids back to a list if needed and append or process as required
        selected_ids_list = list(selected_ids)
        # Example: appending to the primary list, which can be converted back to list if the order is important
        PRIMARY_BUOY_IDS.extend(selected_ids_list)
        config_buoy_ids = PRIMARY_BUOY_IDS
        # Now PRIMARY_BUOY_IDS has the original IDs plus 10 unique IDs from BUOYCAMS2

        # Copy the buoy IDs from the config into the scraper's list
        self.buoycam_ids = list(config_buoy_ids)
        # Scramble the ids
        random.shuffle(self.buoycam_ids)

        # Initialize the tqdm progress bar
        progress_bar = tqdm(total=len(self.buoycam_ids), desc="Initializing")

        for buoycam_id in self.buoycam_ids:
            # Update progress bar description with current buoy ID
            progress_bar.set_description(f"Scraping Buoy ID: {buoycam_id}")
            progress_bar.refresh()  # Refresh the progress bar to show the updated description

            self._scrape_single_buoycam(buoycam_id)

            # Update the progress bar by one step
            progress_bar.update(1)

        # Close the progress bar once all items are processed
        progress_bar.close()


    def _scrape_single_buoycam(self, buoycam_id):
        """
        The _scrape_single_buoycam function takes a buoycam_id as an argument and uses the requests library to retrieve the image from NOAA's website.
        If it is successful, it saves the image using _save_image. If not, it prints a message.

        :param self: Refer to the current instance of the class
        :param buoycam_id: Specify which buoycam to scrape
        :return: The response
        :doc-author: Trelent
        """
        try:
            url = f"https://www.ndbc.noaa.gov/buoycam.php?station={buoycam_id}"
            response = requests.get(url)

            if response.status_code == 200:
                if verbose:
                    print(f"Scraping buoycam {buoycam_id}")
                self._save_image(response.content, buoycam_id)
            else:
                print(f"Failed to retrieve image from buoycam {buoycam_id}")
        except Exception as e:
            logger.error(f"Failed to scrape buoycam {buoycam_id}: {e}")
    def _save_image(self, image_content, buoycam_id):
        """
        The _save_image function takes the image_content and buoycam_id as arguments.
        The timestamp is set to the current time in UTC, formatted as a string with year, month, day, hour minute and second.
        The filename is set to be equal to the buoycam_id plus an underscore plus the timestamp.
        The image path is then created by joining together self (the class), image directory (a variable defined in __init__),
        buoycam id (the argument) and filename (defined above). The os module makes sure that there are no errors if directories already exist.
        Then it opens up a

        :param self: Represent the instance of the class
        :param image_content: Write the image to a file
        :param buoycam_id: Create the directory for each buoycam
        :return: The image_path
        :doc-author: Trelent
        """

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"{buoycam_id}_{timestamp}.jpg"
        image_path = os.path.join(self.image_directory, buoycam_id, filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(image_content)
        if verbose:
            print(f"Image saved: {image_path}")

class ImageProcessor:
    def __init__(self, base_directory):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance variables that are used by other methods in the class.


        :param self: Represent the instance of the class
        :param base_directory: Specify the directory where all of the images will be saved
        :return: The object itself
        :doc-author: Trelent
        """

        self.base_directory = base_directory
        self.panel_directory = os.path.join(base_directory, 'panels')
        os.makedirs(self.panel_directory, exist_ok=True)
        self.latest_images = {}  # Dictionary to hold the latest image per buoy

    def process_images(self):
        """
        The process_images function takes in a list of image files and returns the latest images for each buoy.
            It does this by first creating a dictionary with the buoy_id as key and (file, creation_time) as value.
            Then it iterates through that dictionary to find only the latest images for each buoy.

        :param self: Refer to the object itself, and is used for accessing attributes and methods of the class
        :return: The latest images for each buoy in the panel_directory
        :doc-author: Trelent
        """

        image_files = glob.glob(f'{self.base_directory}/*/*.jpg')
        for file in tqdm(image_files, desc="Processing images"):
            buoy_id = os.path.basename(os.path.dirname(file))
            creation_time = os.path.getctime(file)
            if buoy_id not in self.latest_images or self.latest_images[buoy_id][1] < creation_time:
                self.latest_images[buoy_id] = (file, creation_time) # note: this is a dictionary, so it will only keep the latest image for each buoy
                # we will use this dictionary to process the latest images

        # Now process only the latest images
        ic()
        for buoy_id, (latest_file, _) in self.latest_images.items():
            try:
                image = cv2.imread(latest_file)
                if self._is_valid_image(image):
                    # Enhance the image
                    #note: I have commented out the original enhance images line because I want to fine-tune the way that these images are being processed. The over-enhancements are not looking good.
                    # image = self._enhance_image(image)
                    if verbose:
                        print(f'debug: >> skipped enhancements')
                    #note: this may be reducing size of image, check for this.
                    #!resolved --> this was iCloud uploading the image and reducing file size that appeared to be lower quality, but was actually the same size.

                    # Save the enhanced image
                    cv2.imwrite(os.path.join(self.panel_directory, f"{buoy_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg"), image)
                else:
                    logger.warning(f"buoy {buoy_id}: invalid image")
                    # print(f"Invalid image found for buoy {buoy_id}, skipping")
                    pass
            except Exception as e:
                print(f'Error occurred {e}\n\t buoy id: {buoy_id}')
    def _is_valid_image(self, image, threshold=10):
        """
        The _is_valid_image function is used to determine if an image is valid.
            The function takes in a single argument, the image itself. It then checks
            that the mean of all pixels in the image are within a certain range (10-245).
            If it's not, we assume that there was some sort of error and return False.

        :param self: Allow an object to refer to itself inside of a method
        :param image: Pass in the image that is being tested
        :param threshold: Determine if the image is valid
        :return: A boolean value
        :doc-author: Trelent
        """

        return np.mean(image) >= threshold and np.mean(image) <= 245

    def create_collage_from_latest_images(self):
        """
        The create_collage_from_latest_images function takes the latest images from each buoy and stitches them together into a single image.


        :param self: Refer to the object itself
        :return: A collage of the latest images
        :doc-author: Trelent
        """

        images = []
        for buoy_id, (latest_file, _) in self.latest_images.items():
            image = cv2.imread(latest_file)
            images.append(image)
        return self._stitch_vertical(images)

    def _stitch_vertical(self, rows):
        """
        The _stitch_vertical function takes in a list of images and stitches them together vertically.
        It also checks for duplicate images, black or white images, and resizes the image to fit the max width.

        :param self: Refer to the instance of the class
        :param rows: Pass the list of images that are to be stitched together
        :return: A numpy array of the stiched images
        :doc-author: Trelent
        """

        max_width = max(row.shape[1] for row in rows)
        rows_resized = []
        for row in rows:
            # if the image contains more than the threshold of black pixels or white pixels, skip it
            if np.mean(row) < 10 or np.mean(row) > 245:
                #note: print("Black or white image found, skipping")
                continue
            # if the image is too similar to the previous one, skip it
            if len(rows_resized) > 0 and np.array_equal(row, rows_resized[-1]):
                print("Duplicate image found, skipping")
                continue
            if row.shape[1] < max_width:
                padding = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
                row_resized = np.concatenate((row, padding), axis=1)
            else:
                row_resized = row
            rows_resized.append(row_resized)
        # print the total number of rows to the console
        print(f"Total number of rows: {len(rows_resized)}")
        return np.concatenate(rows_resized, axis=0)

    def _split_into_panels(self, image, number_of_panels=6):
        """
        The _split_into_panels function takes in an image and number of panels as arguments.
        It then splits the image into a list of panels, each one being a numpy array.

        :param self: Refer to the object itself
        :param image: Pass in the image that is being split
        :param number_of_panels: Specify the number of panels to split the image into
        :return: A list of panels
        :doc-author: Trelent
        """

        width = image.shape[1]
        panel_width = width // number_of_panels
        panels = [image[:, i*panel_width:(i+1)*panel_width] for i in range(number_of_panels)]
        # Ensure last panel takes any remaining pixels to account for rounding
        panels[-1] = image[:, (number_of_panels-1)*panel_width:]
        return panels

    def _stitch_panels_horizontally(self, panels):
        """
        The _stitch_panels_horizontally function takes in a list of panels and stitches them together horizontally.

        :param self: Refer to the object itself
        :param panels: Pass in the list of panels that are to be stitched together
        :return: A numpy array of the stitched panels
        :doc-author: Trelent
        """
        # Ensure all panels are the same height before stitching
        max_height = max(panel.shape[0] for panel in panels)
        panels_resized = [cv2.resize(panel, (panel.shape[1], max_height), interpolation=cv2.INTER_LINEAR) for panel in panels]
        return np.concatenate(panels_resized, axis=1)

    def save_collage(self, collage, filename):
        """
        The save_collage function takes in a collage and filename, then saves the collage to the specified file.
            Args:
                self (object): The object that is calling this function.
                collage (numpy array): A numpy array representing an image of a collection of images.
                filename (string): The name of the file where you want to save your image.

        :param self: Allow an object to refer to itself inside of a method
        :param collage: Save the collage image to a file
        :param filename: Save the collage to a specific location
        :return: The filename of the collage
        :doc-author: Trelent
        """

        ic()
        if collage is not None:
            cv2.imwrite(filename, collage)
        else:
            # save the collage also to the temp file so that it can be displayed in the GUI
            # cv2.imwrite("temp.jpg", collage)
            print(f'none type for collage image')
        print(f"Collage saved to {filename}")

    def _enhance_image(self, image):
        """
        Enhance the image by applying modified CLAHE and adjusting color saturation.
            CLAHE Parameters: clipLimit=1.5 and tileGridSize=(8, 8) are used to achieve a balance between enhancing contrast and preventing over-enhancement.
            Saturation Increase: cv2.multiply(image_hsv[:, :, 1], 1.1) increases the saturation channel by 10%, enhancing colors without making them look artificial.
            Error Handling: In case of an error, the original image is returned, and an error message is logged.
        :param self: Refer to the object itself
        :param image: Image to enhance
        :return: Enhanced image
        """
        try:

            # first cut off the bottom 30 pixels if the image is a panorama
            if image.shape[1] > 1000:
                # save those pixels to a separate image which we will reattach later
                bottom_strip = image[-30:, :]
                # remove the bottom strip from the image
                image = image[:-30, :]
            else:
                bottom_strip = None

            # cut the image into 6 panels (horizontally), then process each panel individually
            panels = self._split_into_panels(image, number_of_panels=6)
            processed_panels = []
            for panel in panels:
                try:
                    # # Convert to YUV
                    # image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

                    # # Apply CLAHE to the Y channel (less aggressive settings)
                    # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                    # image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])

                    # # Convert back to BGR
                    # enhanced_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

                    # # Adjust saturation (HSV conversion)
                    # image_hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
                    # image_hsv[:, :, 1] = cv2.multiply(image_hsv[:, :, 1], 1.1) # Increase saturation by 10%
                    # enhanced_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

                    #^ using the above logic for each panel

                    # Convert to YUV
                    panel_yuv = cv2.cvtColor(panel, cv2.COLOR_BGR2YUV)

                    # Apply CLAHE to the Y channel (less aggressive settings)
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                    panel_yuv[:, :, 0] = clahe.apply(panel_yuv[:, :, 0])

                    # Convert back to BGR
                    enhanced_panel = cv2.cvtColor(panel_yuv, cv2.COLOR_YUV2BGR)

                    # Adjust saturation (HSV conversion)
                    panel_hsv = cv2.cvtColor(enhanced_panel, cv2.COLOR_BGR2HSV)
                    panel_hsv[:, :, 1] = cv2.multiply(panel_hsv[:, :, 1], 1.1) # Increase saturation by 10%
                    enhanced_panel = cv2.cvtColor(panel_hsv, cv2.COLOR_HSV2BGR)
                except Exception as e:
                    logger.error(f"Failed to enhance panel: {e}")
                    enhanced_panel = panel

                processed_panels.append(enhanced_panel)

            # Stitch the panels back together
            enhanced_image = self._stitch_panels_horizontally(processed_panels)

            # Reattach the bottom strip if it was removed
            if bottom_strip is not None:
                enhanced_image = np.concatenate((enhanced_image, bottom_strip), axis=0)

        except Exception as e:
            logger.error(f"Failed to enhance image: {e}")
            enhanced_image = image

        return enhanced_image

    #^ patch one: --- Augmentation
    def align_and_trim_panels(self, panels):
        aligned_panels = []
        for panel in panels:
            angle = self._detect_horizon_angle(panel)
            aligned_panel = self._rotate_image(panel, angle)
            trimmed_panel = self._trim_edges(aligned_panel)
            aligned_panels.append(trimmed_panel)
        return aligned_panels

    def _detect_horizon_angle(self, image):
        # Use edge detection and Hough transform to find the most prominent line
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
        # Calculate the angle of this line
        angle = self._calculate_angle(lines)
        return angle

    def _rotate_image(self, image, angle):
        # Rotate the image around its center
        center = (image.shape[1]//2, image.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return rotated_image

    # def _trim_edges(self, image):
    #     # Trim the edges based on a pre-defined trim size or by analyzing the image
    #     trim_size = 10  # For example, trim 10 pixels from each edge
    #     trimmed_image = image[trim_size:-trim_size, trim_size:-trim_size]
    #     return trimmed_image

    def _trim_edges(self, image):
        # Assuming the horizon is now horizontal, determine how much to trim
        # We would need a more sophisticated method to decide how much to trim,
        # possibly based on the standard deviation of the edges or another metric.
        # For now, we will trim a fixed size:

        # Find non-black edge widths
        upper_edge = np.min(np.where(image != 0)[0])
        lower_edge = image.shape[0] - np.max(np.where(image != 0)[0])
        left_edge = np.min(np.where(image != 0)[1])
        right_edge = image.shape[1] - np.max(np.where(image != 0)[1])

        # Trim the black edges
        trimmed_image = image[upper_edge:-lower_edge, left_edge:-right_edge]
        return trimmed_image

    def _calculate_angle(self, lines):
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
            return -np.median(angles)  # Negative because y-coordinates go from top to bottom
        return 0  # No lines found, return 0 degrees


# Main execution
if __name__ == "__main__":
    # IMAGE_DIRECTORY = "images/buoys" #! this is the directory where the images will be saved (OLD)
    IMAGE_DIRECTORY = IMAGES_DIR #* EXPERIMENTAL: this is the directory where the images will be saved (NEW)
    PANEL_DIRECTORY = "panels"

    if not os.path.exists(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
        print(f"Created directory {IMAGE_DIRECTORY}")
    if not os.path.exists(PANEL_DIRECTORY):
        os.makedirs(PANEL_DIRECTORY, exist_ok=True)
        print(f"Created directory {PANEL_DIRECTORY}")

    #* This is a list of all the buoycam ids
    PRIMARY_BUOY_IDS = ["42001","46059","41044","46071","42002","46072","46066","41046","46088","44066","46089","41043","42012","42039","46012","46011","42060","41009","46028","44011","41008","46015","42059","44013","44007","46002","51003","46027","46026","51002","51000","42040","44020","46025","41010","41004","51001","44025","41001","51004","44027","41002","42020","46078","46087","51101","46086","45002","46053","46047","46084","46085","45003","45007","46042","45012","42019","46069","46054","41049","45005","45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084","45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084"]
    # BUOYCAM_IDS = ["21414","21415","21416","21417","21418","21419","32301","32302","32411","32412","32413","41001","41002","41003","41004","41007","41008","41009","41010","41011","41012","41016","41017","41018","41021","41022","41023","41036","41040","41041","41043","41044","41046","41049","41420","41421","41424","41425","42001","42002","42003","42004","42007","42008","42009","42010","42011","42012","42017","42018","42019","42020","42025","42035","42038","42039","42040","42041","42042","42053","42056","42057","42058","42059","42060","42065","42408","42409","42429","42501","42503","42534","44001","44003","44004","44005","44006","44007","44010","44011","44012","44013","44014","44015","44019","44020","44023","44025","44026","44027","44066","44070","44071","44401","44402","44403","45002","45003","45004","45005","45006","45007","45010","45011","45012","46001","46002","46003","46007","46008","46009","46010","46011","46012","46015","46016","46017","46018","46019","46020","46023","46024","46025","46026","46027","46028","46031","46032","46033","46034","46035","46037","46040","46041","46042","46043","46045","46047","46051","46053","46054","46059","46060","46061","46066","46069","46070","46071","46072","46073","46077","46078","46079","46080","46081","46082","46085","46086","46087","46088","46089","46090","46107","46115","46270","46290","46401","46402","46405","46406","46407","46408","46409","46410","46413","46414","46415","46416","46419","46490","46779","46780","46781","46782","46785","51000","51001","51002","51003","51004","51005","51028","51100","51101","51406","51407","51425","52009","52401","52402","52403","52404","52405","91204","91222","91251","91328","91338","91343","91356","91365","91374","91377","91411","91442","46265","41670","41852","41904","41933","48916","48917","52838","52839","52840","52841","52842","52843","52862","55012","55013","55015","55016","55023","55042","58952","31052","31053","41052","41053","41056","41058","41115","41121","41030","44042","44043","44057","44058","44059","44061","44064","44068","45016","45017","45018","45019","45177","45202","45203","45204","45205","45206","45207","46116","46117","46127","42014","42021","42022","42023","42024","42026","32404","41029","41033","41037","41038","41064","41065","41110","41119","41159","32488","41193","44138","44139","44140","44141","44142","44150","44176","44235","44251","44255","44258","44488","45132","45135","45136","45137","45138","45139","45142","45143","45144","45145","45147","45148","45151","45152","45154","45155","45158","45159","46036","46131","46132","46084","46134","46138","46139","46147","46181","46183","46184","46185","46204","46207","46208","46303","46304","48021","45162","45163","45195","23219","23227","32067","32068","42087","42088","42089","42090","46109","46110","46111","46112","21346","21347","21348","21595","21597","21598","21637","21640","22102","22103","22104","22105","22106","22107","45029","45164","45165","45168","45169","45176","46091","46092","62091","62092","62093","62094","41097","41098","41100","41101","41300","61001","45025","45175","44039","44040","44060","23220","23223","23225","46261","46263","48901","48908","48909","48912","44024","44029","44030","44031","44032","44033","44036","44037","45172","45173","46118","46119","46531","46534","46538","46565","44075","44076","44077","44078","46097","46098","51046","51201","51202","51203","51204","51205","51208","51209","51210","51211","51212","51213","52202","52211","13002","13008","13009","13010","15001","15002","31001","31002","31003","31004","31005","31006","62121","62124","62125","62126","62127","62130","62144","62145","62146","62147","62148","62149","62165","62166","63105","63110","63112","63113","14041","14043","14047","23001","23003","23004","23008","23009","23010","23011","23012","23013","23016","23017","53005","53006","53009","53040","56053","01506","01507","01518","01537","48904","48907","01521","01522","01523","01524","01526","01531","01535","01536","01538","01909","01910","31201","41112","41113","41114","41116","41118","41120","42084","42091","42094","42099","44088","44094","44099","44100","44172","46114","46211","46212","46215","46216","46217","46218","46219","46220","46223","46224","46225","46226","46227","46228","46231","46232","46234","46235","46236","46237","46240","46241","46242","46243","46244","46245","46249","46250","46251","46253","46254","46256","46262","46267","46268","46269","46273","46274","51200","48212","48213","48214","48677","48678","48679","48680","48911","42044","42045","42046","42047","42048","42049","42078","42079","42093","42095","42097","44056","45180","46259","46266","62028","62029","62030","62050","62081","62103","62108","62163","62170","62298","62301","62303","62442","64045","44098","46121","46122","46123","46124","28902","28903","28904","28906","28907","28908","58900","58902","58903","58904","58905","58906","58909","68900","78900","45014","45184","44053","01517","32012","41060","41061","21D20","32D12","32D13","41A46","41S43","41S46","46B35","ALSN6","AMAA2","AUGA2","BLIA2","BURL1","BUSL1","CDRF1","CHLV2","CLKN7","CSBF1","DBLN6","DESW1","DRFA2","DRYF1","DSLN7","DUCN7","EB01","EB10","EB33","EB35","EB36","EB43","EB52","EB53","EB70","EB90","EB91","EB92","FARP2","FBIS1","FPSN7","FWYF1","GBCL1","GDIL1","GLLN6","IOSN3","LONF1","LPOI1","MDRM1","MISM1","MLRF1","MPCL1","PILA2","PILM4","PLSF1","POTA2","PTAC1","PTAT2","SANF1","SAUF1","SBIO1","SGNW3","SGOF1","SISW1","SPGF1","SRST2","STDM4","SUPN6","SVLS1","THIN6","VENF1","HBXC1","MYXC1","TDPC1","FSTI2","DMNO3","GPTW1","HMNO3","PRTO3","SEFO3","SETO3","SRAW1","SRFW1","TANO3","ANMF1","ARPF1","BGCF1","CAMF1","CLBF1","EGKF1","NFBF1","PTRF1","SHPF1","MBIN7","MBNN7","OCPN7","BSCA1","CRTA1","DPHA1","KATA1","MBLA1","MHPA1","SACV4","BBSF1","BDVF1","BKYF1","BNKF1","BOBF1","BSKF1","CNBF1","CWAF1","DKKF1","GBIF1","GBTF1","GKYF1","JBYF1","JKYF1","LBRF1","LBSF1","LMDF1","LMRF1","LSNF1","MDKF1","MNBF1","MUKF1","NRRF1","PKYF1","TCVF1","THRF1","TPEF1","TRRF1","WIWF1","WPLF1","APNM4","CHII2","MCYI3","SRLM4","SVNM4","TBIM4","THLO1","LCIY2","LLBP7","FWIC3","MISC3","MISN6","NCSC3","NOSC3","OFPN6","ILDL1","MRSL1","SIPM6","SLPL1","LUML1","TAML1","AKXA2","APMA2","BEXA2","CDXA2","CPXA2","DHXA2","DPXA2","ERXA2","GBXA2","GEXA2","GIXA2","GPXA2","HMSA2","ICYA2","JLXA2","JMLA2","JNGA2","KEXA2","KNXA2","KOZA2","LIXA2","MIXA2","MRNA2","MRYA2","NKLA2","NKXA2","NLXA2","NMXA2","NSXA2","PAUA2","PEXA2","PGXA2","PPXA2","PTLA2","RIXA2","SCXA2","SIXA2","SKXA2","SLXA2","SPXA2","SRXA2","STXA2","SXXA2","TKEA2","TPXA2","UQXA2","VDXA2","WCXA2","MSG10","MSG12","ACQS1","ACXS1","ANMN6","ANRN6","APQF1","APXA2","BILW3","BRIM2","BSLM2","BVQW1","CHNO3","CHQO3","CWQT2","DBQS1","DEQD1","DRSD1","EAZC1","EHSC1","EVMC1","FFFC1","GBHM6","GBQN3","GBRM6","GDQM6","GGGC1","GTQF1","GTXF1","HBMN6","HMRA2","HUQN6","JCTN4","JOBP4","JOQP4","JOXP4","KCHA2","LTQM2","MIST2","MQMT2","MWQT2","NAQR1","NAXR1","NIQS1","NOXN7","NPQN6","NPXN6","OWDO1","OWQO1","OWSO1","PBLW1","PKBW3","RKQF1","RKXF1","RYEC1","SAQG1","SCQC1","SCQN6","SEQA2","SFXC1","SKQN6","SLOO3","TCSV2","TIQC1","TIXC1","TKPN6","WAQM3","WAXM3","WELM1","WEQM1","WEXM1","WKQA1","WKXA1","WYBS1","NLMA3","SBBN2","SLMN2","BAXC1","BDRN4","BDSP1","BGNN6","BKBF1","BLIF1","BRND1","CHCM2","CHYV2","COVM2","CPMW1","CPNW1","CRYV2","DELD1","DMSF1","DOMV2","DPXC1","EBEF1","FMOA1","FRVM3","FRXM3","FSKM2","FSNM2","GCTF1","LNDC1","LQAT2","LTJF1","MBPA1","MCGA1","MHBT2","MRCP1","MTBF1","MZXC1","NBLP1","NFDF1","NWHC3","OMHC1","OPTF1","PDVR1","PEGF1","PFDC1","PFXC1","PPTM2","PPXC1","PRJC1","PRUR1","PSBC1","PSXC1","PTOA1","PVDR1","PXAC1","PXOC1","PXSC1","QPTR1","RPLV2","RTYC1","SEIM1","SJSN4","SKCF1","SWPM4","TCNW1","TLVT2","TPAF1","TSHF1","TXVT2","UPBC1","WDSV2","ACYN4","ADKA2","AGCM4","ALIA2","ALXN6","AMRL1","APAM2","APCF1","APRP7","ASTO3","ATGM1","ATKA2","BEPB6","BFTN7","BHBM3","BISM2","BKTL1","BLTM2","BYGL1","BZBM3","CAMM2","CAPL1","CARL1","CASM1","CECC1","CFWM1","CHAO3","CHAV3","CHBV2","CHSV3","CHYW1","CLBP4","CMAN4","CMTI2","CNDO1","CRVA2","DILA1","DKCM6","DTLM4","DUKN7","DULM5","EBSW1","ERTF1","ESPP4","FAIO1","FCGT2","FMRF1","FOXR1","FPTT2","FRCB6","FRDF1","FRDW1","FREL1","FRPS1","FTPC1","GBWW3","GCVF1","GDMM5","GISL1","GNJT2","GTOT2","GWPM6","HBYC1","HCGN7","HLNM4","HMDO3","ICAC1","IIWC1","ILOH1","ITKA2","JMPN7","JNEA2","KECA2","KGCA2","KLIH1","KPTN6","KPTV2","KWHH1","KYWF1","LABL1","LAMV3","LAPW1","LCLL1","LDTM4","LOPW1","LPNM4","LTBV3","LTRM4","LWSD1","LWTV2","MBRM4","MCGM4","MCYF1","MEYC1","MGIP4","MGZP4","MOKH1","MQTT2","MRHO1","MROS1","MTKN6","MTYC1","NEAW1","NIAN6","NJLC1","NKTA2","NLNC3","NMTA2","NTBC1","NTKM3","NUET2","NWCL1","NWPR1","NWWH1","OCIM2","OHBC1","OLSA2","OOUH1","ORIN7","OSGN6","PCBF1","PCLF1","PCOC1","PGBP7","PHBP1","PLXA2","PNLM6","PORO3","PRDA2","PRYC1","PSBM1","PSLC1","PTAW1","PTIM4","PTIT2","PTWW1","RARM6","RCKM4","RCYF1","RDDA2","RDYD1","SAPF1","SBEO3","SBLF1","SDBC1","SDHN4","SHBL1","SJNP4","SKTA2","SLIM2","SNDP5","SWLA2","SWPV2","TESL1","THRO1","TLBO3","TRDF1","TXPT2","ULAM6","ULRA2","UNLA2","VAKF1","VDZA2","WAHV2","WAKP8","WASD2","WAVM6","WLON7","WPTW1","WYCM6","YATA2","BLTA2","CDEA2","EROA2","LCNA2","PBPA2","PRTA2","SDIA2","AGMW3","BHRI3","BIGM4","BSBM4","CBRW3","CLSM4","FPTM4","GBLW3","GRMM4","GSLM4","GTLM4","GTRM4","KP53","KP58","KP59","LSCM4","MEEM4","NABM4","PCLM4","PNGW3","PRIM4","PSCM4","PWAW3","SBLM4","SPTM4","SXHW3","SYWW3","TAWM4","WFPM4","BARN6","CBLO1","CHDS1","CMPO1","GELO1","HHLO1","LORO1","NREP1","OLCN6","RPRN6","WATS1","AUDP4","FRDP4","PLSP4","VQSP4","CGCL1","SKMG1","SPAG1","AVAN4","BRBN4","OCGN4","AWRT2","BABT2","BZST2","CLLT2","CPNT2","EMAT2","GRRT2","HIST2","IRDT2","LUIT2","LYBT2","MGPT2","NWST2","PACT2","PCGT2","PCNT2","PMNT2","PORT2","RSJT2","RTAT2","RTOT2","SDRT2","SGNT2","TAQT2","BTHD1","FRFN7","JPRN7","18CI3","20CM4","GDIV2","32ST0","41NT0"]

    # open pybuoys_

    # BUOYCAM_IDS = ["42001", "46059", "41044", "46071", "42002", "46072", "46066", "41046", "46088", "44066", "46089", "41043", "42012", "42039", "46012", "46011", "42060", "41009", "46028", "44011", "41008", "46015", "42059", "44013", "44007", "46002", "51003", "46027", "46026", "51002", "51000", "42040", "44020", "46025", "41010", "41004", "51001", "44025", "41001", "51004", "44027", "41002", "42020", "46078", "46087", "51101", "46086", "45002", "46053", "46047", "46084", "46085", "45003", "45007", "46042", "45012", "42019", "46069", "46054", "41049", "45005"]


    # remove dupes
    BUOYCAM_IDS = PRIMARY_BUOY_IDS
    BUOYCAM_IDS = list(set(BUOYCAM_IDS))
    scraper = BuoyCamScraper(IMAGE_DIRECTORY, BUOYCAM_IDS)


    # Scrape images from each buoycam starting at 4 am on the East Coast (9 am UTC) and ending at 9 am on the East Coast (2 pm UTC) every 15 minutes
    # Scrape the west coast for sunset starting at 4 PM on the West Coast and ending at 9 PM on the West Coast every 15 minutes

    INTERVAL = 15 #^ This is the interval in minutes
    START_HOUR = 4 #^ This is the start hour in UTC time
    END_HOUR = 23 #^ This is the end hour in UTC time
    while True:
        try:

            # current_hour = datetime.utcnow().hour
            # if the time is between 4 am and 9 am on the East Coast (9 am and 2 pm UTC), scrape the images
            ic()
            # print(f'Current Hour: {current_hour}')
            # if current_hour >= START_HOUR and current_hour <= END_HOUR:
            if bypassing_hour or datetime.utcnow().hour >= START_HOUR and datetime.utcnow().hour <= END_HOUR:
                scraper.scrape()
                ic()
                try:
                    if verbose:
                        print(f'Trying to process images...')
                    processor = ImageProcessor(IMAGE_DIRECTORY)
                    processor.process_images()  # This will now only process the latest images
                    ic()
                    # Stitching the latest images into a collage
                    collage = processor.create_collage_from_latest_images()
                    processor.save_collage(collage, f"images/save_images/collage_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg")

                except Exception as e:
                    print(f'I ran into an error!\n\t {e}')
                time_beforecleaning = datetime.utcnow().strftime('%Y%m%d%H%M%S')
                if CLEANING_ACTIVE:
                    # remove white images
                    remove_whiteimages(IMAGE_DIRECTORY) # will remove white images from the images/buoys directory ON THE HARD DRIVE
                    print(f'White images removed from images/buoys directory')
                    remove_similar_images('images/buoys') # will remove similar images from the images/buoys directory
                    print(f'Similar or Duplicated images removed from images/buoys directory')

                    try:
                        # remove lines from the top of the log file until it is under 10 MB
                        size_log_file = os.path.getsize('logs/pybuoy.log')
                        while size_log_file > 10000000:
                            with open('logs/pybuoy.log', 'r') as log_file:
                                lines = log_file.readlines()
                            with open('logs/pybuoy.log', 'w') as log_file:
                                log_file.writelines(lines[1:])
                            size_log_file = os.path.getsize('logs/pybuoy.log')
                        print(f'Log file cleaned')
                    except Exception as e:
                        print(f'Error cleaning log file: {e}') #todo -- this is beta, test it and make sure it works
                #* this is the time after cleaning
                time_aftercleaning = datetime.utcnow().strftime('%Y%m%d%H%M%S')
                # convert to a number
                time_beforecleaning = float(time_beforecleaning)
                time_aftercleaning = float(time_aftercleaning)
                # calculate the time delta
                time_delta = time_aftercleaning - time_beforecleaning
                time_delta = time_delta / 60 # convert to minutes
                # we want to wait for the remainder of the interval before continuing to the next iteration of the loop which is the interval - the time it took to clean the images
                time_delta = int(round(time_delta))
                if time_delta < 0:
                    time_delta = 0 # initialize the time delta to 0 if it is negative
                #* this is the time it took to clean the images
                print(f'Sleeping for {INTERVAL * 60 - time_delta} seconds...')
                #todo -- the comments below could be useful. I am not sure if I want to print the IDs of the buoys that are still showing images or not.
                #? print the IDs of the Buoys that are still showing images (in the collage)
                #? print(f'Buoy IDs in the collage: {processor.latest_images.keys()}')
                #?logger.info(f'Buoy IDs in the collage: {processor.latest_images.keys()}')
                for i in tqdm(range(0, INTERVAL * 60 - time_delta)):
                    time.sleep(1) # sleep for 1 second
            else:
                print(f'Waiting until {START_HOUR} UTC to start scraping...')
                for i in tqdm(range(0, 60)):
                    time.sleep(1)
                if datetime.utcnow().hour >= START_HOUR:
                    bypassing_hour = True # todo -- this is a crude method of bypassing the hour, but it works for now. Fix this later.
                    print(f'Starting to scrape...')
        except Exception as e:
            print(f'I ran into an error!\n\t {e}')
            logger.error("%s\n\tError in main loop, waiting one minute before continuing...", e)
            time.sleep(60)
            #note: this keeps the script from crashing if there is an error
            #todo -- add a way to restart the script if it crashes
