import glob
import logging
import logging.handlers  # this is for the rotating file handler which means that the log file will not get too large and will be easier to read
import os
import sys
import time
from datetime import datetime

import cv2
import icecream
import numpy as np
import requests
from icecream import ic
from tqdm import tqdm

from white_image_removal import remove_whiteimages
# from duplicate_removal import remove_duplicates, remove_duplicates_from_csv
from duplicates import remove_similar_images #* this works best, requires a root directory to be specified
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
    open('logs/pybuoy.log', 'a').close()

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
IMAGES_DIR = '/Volumes/THEVAULT/IMAGES' #^ This is the directory where the images will be saved
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

    def scrape(self):
        """
        The scrape function is the main function of this module. It takes a list of buoycam_ids and scrapes each one individually,
            using the _scrape_single_buoycam function. The scrape function also handles any errors that may occur during scraping.

        :param self: Refer to the object itself
        :return: The data from the buoycam_ids
        :doc-author: Trelent
        """

        for buoycam_id in self.buoycam_ids:
            self._scrape_single_buoycam(buoycam_id)

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
        cv2.imwrite(filename, collage)
        # save the collage also to the temp file so that it can be displayed in the GUI
        cv2.imwrite("temp.jpg", collage)
        print(f"Collage saved to {filename} and to the GUI file temp.jpg")

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


# Main execution
if __name__ == "__main__":
    IMAGE_DIRECTORY = "images/buoys" #! this is the directory where the images will be saved (OLD)
    IMAGE_DIRECTORY = IMAGES_DIR #* EXPERIMENTAL: this is the directory where the images will be saved (NEW)
    PANEL_DIRECTORY = "panels"

    if not os.path.exists(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
        print(f"Created directory {IMAGE_DIRECTORY}")
    if not os.path.exists(PANEL_DIRECTORY):
        os.makedirs(PANEL_DIRECTORY, exist_ok=True)
        print(f"Created directory {PANEL_DIRECTORY}")

    #* This is a list of all the buoycam ids
    # BUOYCAM_IDS = ["42001","46059","41044","46071","42002","46072","46066","41046","46088","44066","46089","41043","42012","42039","46012","46011","42060","41009","46028","44011","41008","46015","42059","44013","44007","46002","51003","46027","46026","51002","51000","42040","44020","46025","41010","41004","51001","44025","41001","51004","44027","41002","42020","46078","46087","51101","46086","45002","46053","46047","46084","46085","45003","45007","46042","45012","42019","46069","46054","41049","45005","45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084","45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084"]
    BUOYCAM_IDS = ["42001", "46059", "41044", "46071", "42002", "46072", "46066", "41046", "46088", "44066", "46089", "41043", "42012", "42039", "46012", "46011", "42060", "41009", "46028", "44011", "41008", "46015", "42059", "44013", "44007", "46002", "51003", "46027", "46026", "51002", "51000", "42040", "44020", "46025", "41010", "41004", "51001", "44025", "41001", "51004", "44027", "41002", "42020", "46078", "46087", "51101", "46086", "45002", "46053", "46047", "46084", "46085", "45003", "45007", "46042", "45012", "42019", "46069", "46054", "41049", "45005"]


    # remove dupes
    BUOYCAM_IDS = list(set(BUOYCAM_IDS))
    scraper = BuoyCamScraper(IMAGE_DIRECTORY, BUOYCAM_IDS)


    # Scrape images from each buoycam starting at 4 am on the East Coast (9 am UTC) and ending at 9 am on the East Coast (2 pm UTC) every 15 minutes
    # Scrape the west coast for sunset starting at 4 PM on the West Coast and ending at 9 PM on the West Coast every 15 minutes

    INTERVAL = 15 #^ This is the interval in minutes
    START_HOUR = 4 #^ This is the start hour in UTC time
    END_HOUR = 24 #^ This is the end hour in UTC time
    while True:
        try:

            # current_hour = datetime.utcnow().hour
            # if the time is between 4 am and 9 am on the East Coast (9 am and 2 pm UTC), scrape the images
            ic()
            # print(f'Current Hour: {current_hour}')
            # if current_hour >= START_HOUR and current_hour <= END_HOUR:
            if bypassing_hour:
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
