from datetime import datetime
from scraper import scrape_single_buoycam
from image_processor import logger, process_images
import concurrent.futures
from tqdm import tqdm
import time
from config import INTERVAL, BUOYCAM_IDS


def main():
    while True:
        current_hour = datetime.utcnow().hour
        try:
            # get the Inverval and buoy ids again from config.py
            # from config import INTERVAL, BUOYCAM_IDS
            # print(f'loaded updated config.py with INTERVAL={INTERVAL} and BUOYCAM_IDS={BUOYCAM_IDS}')
            if 4 <= current_hour <= 24:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(scrape_single_buoycam, BUOYCAM_IDS)
                    #& results is a list of results from the scrape_single_buoycam function
                    for result, buoycam_id in tqdm(zip(results, BUOYCAM_IDS)):
                        if result:
                            #^ if result is not None meaning it is a valid image
                            executor.submit(process_images,
                                            result,
                                            buoycam_id)
                        else: #& if result is None meaning it is an invalid image
                            logger.warning(f'buoy {buoycam_id}: invalid image')
                print(f'Sleeping for {INTERVAL} minutes')
                for i in tqdm(range(0, INTERVAL * 60)):
                    time.sleep(1) # sleep for 1 second
            else:
                for i in tqdm(range(0, 60)):
                    time.sleep(1)
        except Exception as e:
            #& Log the error
            print(f'Error: {e}')
            logger.error(f'Error: {e}')
if __name__ == "__main__":
    main()
