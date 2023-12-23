import glob
import logging
import logging.handlers
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
from duplicates import remove_similar_images
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('logs/pybuoy.log'):
    open('logs/pybuoy.log', 'a').close()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler('logs/pybuoy.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
verbose = False
bypassing_hour = True
CLEANING_ACTIVE = True
IMAGES_DIR = '/Users/grahamwaters/Library/CloudStorage/GoogleDrive-graham.waters.business@gmail.com/My Drive/pyseas_images'
class BuoyCamScraper:
    def __init__(self, image_directory, buoycam_ids):
        self.image_directory = image_directory
        self.buoycam_ids = buoycam_ids
        os.makedirs(self.image_directory, exist_ok=True)
    def scrape(self):
        for buoycam_id in self.buoycam_ids:
            self._scrape_single_buoycam(buoycam_id)
    def _scrape_single_buoycam(self, buoycam_id):
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
        self.base_directory = base_directory
        self.panel_directory = os.path.join(base_directory, 'panels')
        os.makedirs(self.panel_directory, exist_ok=True)
        self.latest_images = {}
    def process_images(self):
        image_files = glob.glob(f'{self.base_directory}/*/*.jpg')
        for file in tqdm(image_files, desc="Processing images"):
            buoy_id = os.path.basename(os.path.dirname(file))
            creation_time = os.path.getctime(file)
            if buoy_id not in self.latest_images or self.latest_images[buoy_id][1] < creation_time:
                self.latest_images[buoy_id] = (file, creation_time)
        ic()
        for buoy_id, (latest_file, _) in self.latest_images.items():
            image = cv2.imread(latest_file)
            if self._is_valid_image(image):
                if verbose:
                    print(f'debug: >> skipped enhancements')
                cv2.imwrite(os.path.join(self.panel_directory, f"{buoy_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg"), image)
            else:
                logger.warning(f"buoy {buoy_id}: invalid image")
                pass
    def _is_valid_image(self, image, threshold=10):
        return np.mean(image) >= threshold and np.mean(image) <= 245
    def create_collage_from_latest_images(self):
        images = []
        for buoy_id, (latest_file, _) in self.latest_images.items():
            image = cv2.imread(latest_file)
            images.append(image)
        return self._stitch_vertical(images)
    def _stitch_vertical(self, rows):
        max_width = max(row.shape[1] for row in rows)
        rows_resized = []
        for row in rows:
            if np.mean(row) < 10 or np.mean(row) > 245:
                continue
            if len(rows_resized) > 0 and np.array_equal(row, rows_resized[-1]):
                print("Duplicate image found, skipping")
                continue
            if row.shape[1] < max_width:
                padding = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
                row_resized = np.concatenate((row, padding), axis=1)
            else:
                row_resized = row
            rows_resized.append(row_resized)
        print(f"Total number of rows: {len(rows_resized)}")
        return np.concatenate(rows_resized, axis=0)
    def _split_into_panels(self, image, number_of_panels=6):
        width = image.shape[1]
        panel_width = width // number_of_panels
        panels = [image[:, i*panel_width:(i+1)*panel_width] for i in range(number_of_panels)]
        panels[-1] = image[:, (number_of_panels-1)*panel_width:]
        return panels
    def _stitch_panels_horizontally(self, panels):
        max_height = max(panel.shape[0] for panel in panels)
        panels_resized = [cv2.resize(panel, (panel.shape[1], max_height), interpolation=cv2.INTER_LINEAR) for panel in panels]
        return np.concatenate(panels_resized, axis=1)
    def save_collage(self, collage, filename):
        ic()
        cv2.imwrite(filename, collage)
        cv2.imwrite("temp.jpg", collage)
        print(f"Collage saved to {filename} and to the GUI file temp.jpg")
    def _enhance_image(self, image):
        try:
            if image.shape[1] > 1000:
                bottom_strip = image[-30:, :]
                image = image[:-30, :]
            else:
                bottom_strip = None
            panels = self._split_into_panels(image, number_of_panels=6)
            processed_panels = []
            for panel in panels:
                try:
                    panel_yuv = cv2.cvtColor(panel, cv2.COLOR_BGR2YUV)
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                    panel_yuv[:, :, 0] = clahe.apply(panel_yuv[:, :, 0])
                    enhanced_panel = cv2.cvtColor(panel_yuv, cv2.COLOR_YUV2BGR)
                    panel_hsv = cv2.cvtColor(enhanced_panel, cv2.COLOR_BGR2HSV)
                    panel_hsv[:, :, 1] = cv2.multiply(panel_hsv[:, :, 1], 1.1)
                    enhanced_panel = cv2.cvtColor(panel_hsv, cv2.COLOR_HSV2BGR)
                except Exception as e:
                    logger.error(f"Failed to enhance panel: {e}")
                    enhanced_panel = panel
                processed_panels.append(enhanced_panel)
            enhanced_image = self._stitch_panels_horizontally(processed_panels)
            if bottom_strip is not None:
                enhanced_image = np.concatenate((enhanced_image, bottom_strip), axis=0)
        except Exception as e:
            logger.error(f"Failed to enhance image: {e}")
            enhanced_image = image
        return enhanced_image
if __name__ == "__main__":
    IMAGE_DIRECTORY = IMAGES_DIR
    PANEL_DIRECTORY = "panels"
    if not os.path.exists(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
        print(f"Created directory {IMAGE_DIRECTORY}")
    if not os.path.exists(PANEL_DIRECTORY):
        os.makedirs(PANEL_DIRECTORY, exist_ok=True)
        print(f"Created directory {PANEL_DIRECTORY}")
    BUOYCAM_IDS = ["42001","46059","41044","46071","42002","46072","46066","41046","46088","44066","46089","41043","42012","42039","46012","46011","42060","41009","46028","44011","41008","46015","42059","44013","44007","46002","51003","46027","46026","51002","51000","42040","44020","46025","41010","41004","51001","44025","41001","51004","44027","41002","42020","46078","46087","51101","46086","45002","46053","46047","46084","46085","45003","45007","46042","45012","42019","46069","46054","41049","45005","45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084","45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084"]
    BUOYCAM_IDS = list(set(BUOYCAM_IDS))
    scraper = BuoyCamScraper(IMAGE_DIRECTORY, BUOYCAM_IDS)
    INTERVAL = 15
    START_HOUR = 4
    END_HOUR = 23
    while True:
        try:
            ic()
            if bypassing_hour or datetime.utcnow().hour >= START_HOUR and datetime.utcnow().hour <= END_HOUR:
                scraper.scrape()
                ic()
                try:
                    if verbose:
                        print(f'Trying to process images...')
                    processor = ImageProcessor(IMAGE_DIRECTORY)
                    processor.process_images()
                    ic()
                    collage = processor.create_collage_from_latest_images()
                    processor.save_collage(collage, f"images/save_images/collage_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg")
                except Exception as e:
                    print(f'I ran into an error!\n\t {e}')
                time_beforecleaning = datetime.utcnow().strftime('%Y%m%d%H%M%S')
                if CLEANING_ACTIVE:
                    remove_whiteimages(IMAGE_DIRECTORY)
                    print(f'White images removed from images/buoys directory')
                    remove_similar_images('images/buoys')
                    print(f'Similar or Duplicated images removed from images/buoys directory')
                    try:
                        size_log_file = os.path.getsize('logs/pybuoy.log')
                        while size_log_file > 10000000:
                            with open('logs/pybuoy.log', 'r') as log_file:
                                lines = log_file.readlines()
                            with open('logs/pybuoy.log', 'w') as log_file:
                                log_file.writelines(lines[1:])
                            size_log_file = os.path.getsize('logs/pybuoy.log')
                        print(f'Log file cleaned')
                    except Exception as e:
                        print(f'Error cleaning log file: {e}')
                time_aftercleaning = datetime.utcnow().strftime('%Y%m%d%H%M%S')
                time_beforecleaning = float(time_beforecleaning)
                time_aftercleaning = float(time_aftercleaning)
                time_delta = time_aftercleaning - time_beforecleaning
                time_delta = time_delta / 60
                time_delta = int(round(time_delta))
                if time_delta < 0:
                    time_delta = 0
                print(f'Sleeping for {INTERVAL * 60 - time_delta} seconds...')
                for i in tqdm(range(0, INTERVAL * 60 - time_delta)):
                    time.sleep(1)
            else:
                print(f'Waiting until {START_HOUR} UTC to start scraping...')
                for i in tqdm(range(0, 60)):
                    time.sleep(1)
                if datetime.utcnow().hour >= START_HOUR:
                    bypassing_hour = True
                    print(f'Starting to scrape...')
        except Exception as e:
            print(f'I ran into an error!\n\t {e}')
            logger.error("%s\n\tError in main loop, waiting one minute before continuing...", e)
            time.sleep(60)
