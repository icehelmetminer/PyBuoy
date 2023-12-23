# The refactored `BuoyCamScraper` script is now complete with all the original logic integrated. The script is organized into a class-based structure, making it more maintainable and readable. Here's an overview of the script:

# 1. **Class BuoyCamScraper**: This class handles the main functionality of scraping buoy camera images. It includes methods for scraping all buoys (`scrape_all_buoys`), scraping a single buoy camera (`_scrape_single_buoycam`), and saving images (`_save_image`).

# 2. **Scheduling with APScheduler**: The script uses `APScheduler` to schedule the scraping task. This replaces the original continuous loop and `time.sleep()` approach, providing a more robust and efficient way to handle periodic tasks.

# 3. **Post-Processing**: After scraping the images, the script performs post-processing which includes removing white images and similar or duplicated images.

# 4. **Configuration Variables**: The script imports configuration variables from `config.py`, centralizing the configuration and making it easy to adjust settings.

# 5. **Logging**: The script includes logging for error tracking and debugging purposes.

# This refactored script aligns with the requirements of using a class-based structure, moving variables to a configuration file, and implementing a scheduling feature for the processes. The script is also more readable and maintainable, and it's easier to add new features and functionality.

import logging
import os
import requests
from datetime import datetime
import cv2
from tqdm import tqdm
from white_image_removal import remove_whiteimages
from duplicates import remove_similar_images
from apscheduler.schedulers.blocking import BlockingScheduler
import config

class BuoyCamScraper:
    def __init__(self):
        self.image_directory = config.IMAGE_DIRECTORY
        self.panel_directory = config.PANEL_DIRECTORY
        self.collage_directory = config.COLLAGE_DIRECTORY
        self.buoycam_ids = config.BUOYCAM_IDS
        self.interval = config.INTERVAL
        os.makedirs(self.image_directory, exist_ok=True)

    def scrape(self):
        scheduler = BlockingScheduler()
        scheduler.add_job(self.scrape_all_buoys, 'interval', minutes=self.interval)
        scheduler.start()

    def scrape_all_buoys(self):
        for buoycam_id in self.buoycam_ids:
            self._scrape_single_buoycam(buoycam_id)
        self._post_process_images()

    # Integrated _scrape_single_buoycam logic
    def _scrape_single_buoycam(self, buoycam_id):
        buoycam_url = f"https://www.ndbc.noaa.gov/data/buoycam/{buoycam_id}.jpg"
        response = requests.get(buoycam_url)
        if response.status_code == 200:
            image_name = f"{buoycam_id}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.jpg"
            image_path = os.path.join(self.image_directory, image_name)
            with open(image_path, "wb") as f:
                f.write(response.content)
            self._save_image(image_path, image_name)
        else:
            logging.error(f"Failed to scrape buoycam {buoycam_id}")


    # Integrated _save_image logic
    def _save_image(self, image_path, image_name):
        image = cv2.imread(image_path)
        panel_image_path = os.path.join(self.panel_directory, image_name)
        collage_image_path = os.path.join(self.collage_directory, image_name)
        cv2.imwrite(panel_image_path, image)
        cv2.imwrite(collage_image_path, image)

    # Integrated _post_process_images logic
    def _post_process_images(self):
        remove_whiteimages(self.panel_directory)
        # remove_similar_images(self.panel_directory)
        remove_similar_images(self.collage_directory)


if __name__ == "__main__":
    scraper = BuoyCamScraper()
    scraper.scrape()
