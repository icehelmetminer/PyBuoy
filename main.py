from datetime import datetime
from scraper import scrape_single_buoycam
from image_processor import process_images
from config import BUOYCAM_IDS, INTERVAL
import concurrent.futures
from tqdm import tqdm
import time

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
