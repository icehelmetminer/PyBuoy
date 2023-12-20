import cv2
import numpy as np
import glob
import os
import requests
from datetime import datetime
from tqdm import tqdm

## Author: Graham Waters
## Date: 12/17/2023
## Description: This script scrapes the NOAA buoycam images and creates a collage of the latest images from each buoycam. It also saves the latest images for each buoycam in a separate directory. The beauty of the worlds oceans is captured in these images, and this script allows you to view them all in one place.

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

        url = f"https://www.ndbc.noaa.gov/buoycam.php?station={buoycam_id}"
        response = requests.get(url)
        if response.status_code == 200:
            self._save_image(response.content, buoycam_id)
        else:
            print(f"Failed to retrieve image from buoycam {buoycam_id}")

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
                self.latest_images[buoy_id] = (file, creation_time)

        # Now process only the latest images
        for buoy_id, (latest_file, _) in self.latest_images.items():
            image = cv2.imread(latest_file)
            if self._is_valid_image(image):
                cv2.imwrite(os.path.join(self.panel_directory, f"{buoy_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg"), image)
            else:
                print(f"Invalid image found for buoy {buoy_id}, skipping")
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
                print("Black or white image found, skipping")
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
        return np.concatenate(rows_resized, axis=0)

    def save_collage(self, collage, filename):
        cv2.imwrite(filename, collage)
        print(f"Collage saved to {filename}")

# Main execution
if __name__ == "__main__":
    IMAGE_DIRECTORY = "images/buoys"
    PANEL_DIRECTORY = "panels"
    BUOYCAM_IDS = ["45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084"]

    scraper = BuoyCamScraper(IMAGE_DIRECTORY, BUOYCAM_IDS)
    scraper.scrape()

    processor = ImageProcessor(IMAGE_DIRECTORY)

    processor.process_images()  # This will now only process the latest images

    # Stitching the latest images into a collage
    collage = processor.create_collage_from_latest_images()
    processor.save_collage(collage, f"images/save_images/collage_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg")