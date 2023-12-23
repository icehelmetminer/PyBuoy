from scraper import BuoyCamScraper
from image_processor import ImageProcessor
from utils import remove_whiteimages, remove_similar_images
from logger import setup_logging
import os
from datetime import datetime

def schedule_all_tasks(scheduler):
    # Set up logging
    logger = setup_logging()

    # Initialize classes
    scraper = BuoyCamScraper(os.path.join('images', 'buoys'), buoycam_ids)
    processor = ImageProcessor(os.path.join('images', 'buoys'))

    def scrape_task():
        logger.info("Starting scraping task.")
        scraper.scrape()
        logger.info("Scraping task completed.")

    def process_images_task():
        logger.info("Starting image processing task.")
        processor.process_images()
        logger.info("Image processing task completed.")

    def cleanup_task():
        logger.info("Starting cleanup task.")
        remove_whiteimages()
        remove_similar_images()
        logger.info("Cleanup task completed.")

    # Schedule tasks
    scheduler.schedule_task(scrape_task, interval_minutes=15)
    scheduler.schedule_task(process_images_task, interval_minutes=20)
    scheduler.schedule_task(cleanup_task, interval_minutes=30)
