import requests
from config import BUOYCAM_IDS
from logger import setup_logger

logger = setup_logger()

def scrape_single_buoycam(buoycam_id):
    try:
        url = f"https://www.ndbc.noaa.gov/buoycam.php?station={buoycam_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"Failed to retrieve image from buoycam {buoycam_id}")
            return None
    except Exception as e:
        logger.error(f"Failed to scrape buoycam {buoycam_id}: {e}")
        return None
