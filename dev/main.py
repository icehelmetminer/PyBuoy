import threading
import schedule
import time
from image_processor import ImageProcessor
from scraper import BuoyCamScraper
import logging

# Setting up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
file_handler = logging.FileHandler('logs/main.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

from pybuoy_final import scrape_images, process_images

#^ Setting up constants
# IMAGE_DIRECTORY = '/Volumes/THEVAULT/IMAGES'
# BUOYCAM_IDS = ['42001', '46059']


def safe_execute(function):
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {function.__name__}: {e}")
            return None
    return wrapper

# Function to scrape buoy images
@safe_execute
def scrape_images():
    scraper = BuoyCamScraper(verbose=True)
    scraper.scrape()

# Function to process images
def process_images():
    processor = ImageProcessor(IMAGE_DIRECTORY, verbose=True)
    processor.process_images()

# Scheduling tasks
schedule.every(15).minutes.do(scrape_images)
schedule.every(30).minutes.do(process_images)

# Running the scheduled tasks in a separate thread
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    # Starting the scheduling thread
    threading.Thread(target=run_schedule, daemon=True).start()

    # Main loop
    try:
        while True:
            # Main script can perform other tasks or remain idle
            time.sleep(60)
    except KeyboardInterrupt:
        print("Script interrupted and stopped.")
