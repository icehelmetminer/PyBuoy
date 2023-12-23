import requests
import os
from datetime import datetime
from icecream import ic
from config import BUOYCAM_IDS, INTERVAL

IMAGE_DIRECTORY = '/Volumes/THEVAULT/IMAGES'


class BuoyCamScraper:
    def __init__(self, verbose=False):
        self.image_directory = '/Volumes/THEVAULT/IMAGES'
        self.buoycam_ids = ["42001","46059","41044","46071","42002","46072","46066","41046","46088","44066","46089","41043","42012","42039","46012","46011","42060","41009","46028","44011","41008","46015","42059","44013","44007","46002","51003","46027","46026","51002","51000","42040","44020","46025","41010","41004","51001","44025","41001","51004","44027","41002","42020","46078","46087","51101","46086","45002","46053","46047","46084","46085","45003","45007","46042","45012","42019","46069","46054","41049","45005","45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084","45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084"]
        self.verbose = True
        os.makedirs(self.image_directory, exist_ok=True) # exist okay means it won't throw an error if the directory already exists, it will just continue

    def _scrape_single_buoycam(self, buoycam_id):
        try:
            url = f"https://www.ndbc.noaa.gov/buoycam.php?station={buoycam_id}"
            response = requests.get(url)
            if response.status_code == 200:
                self._save_image(response.content, buoycam_id)
                if self.verbose:
                    ic(f"Image scraped: {buoycam_id}")
            else:
                if self.verbose:
                    ic(f"Failed to retrieve image from {buoycam_id}")
        except Exception as e:
            if self.verbose:
                ic(e)

    def _save_image(self, image_content, buoycam_id):
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"{buoycam_id}_{timestamp}.jpg"
        image_path = os.path.join(self.image_directory, buoycam_id, filename)
        with open(image_path, "wb") as file:
            file.write(image_content)
        if self.verbose:
            ic(f"Image saved: {image_path}")

    def scrape(self):
        for buoycam_id in self.buoycam_ids:
            self._scrape_single_buoycam(buoycam_id)

# Example usage
if __name__ == "__main__":
    scraper = BuoyCamScraper()
    scraper.scrape()
