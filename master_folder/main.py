import scheduled_tasks
from scheduler import Scheduler
from scraper import BuoyCamScraper, scrape_single_buoycam
from image_processor import ImageProcessor
from utils import remove_whiteimages, remove_similar_images
from logger import setup_logging, setup_logger
import os
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import time
from config import BUOYCAM_IDS, INTERVAL, IMAGE_DIRECTORY


def schedule_all_tasks(scheduler):
    # Set up logging
    logger = setup_logger()

    # Initialize classes
    scraper = BuoyCamScraper(IMAGE_DIRECTORY, BUOYCAM_IDS)

    def scrape_task():
        logger.info("Starting scraping task.")
        scraper.scrape()
        logger.info("Scraping task completed.")

    # Schedule tasks
    scheduler.schedule_task(scrape_task, interval_minutes=INTERVAL)

def main():
    while True:
        current_hour = datetime.utcnow().hour
        if 4 <= current_hour <= 24:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(scrape_single_buoycam, BUOYCAM_IDS)
                for result, buoycam_id in zip(results, BUOYCAM_IDS):
                    if result:
                        executor.submit(process_images, result, buoycam_id)
            for i in tqdm(range(0, INTERVAL * 60)):
                time.sleep(1) # sleep for 1 second
        else:
            for i in tqdm(range(0, 60)):
                time.sleep(1)


if __name__ == "__main__":
    main()
