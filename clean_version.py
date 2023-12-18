import cv2
import numpy as np
import glob
import os
import requests
from datetime import datetime
from tqdm import tqdm

class BuoyCamScraper:
    def __init__(self, image_directory, buoycam_ids):
        self.image_directory = image_directory
        self.buoycam_ids = buoycam_ids
        os.makedirs(self.image_directory, exist_ok=True)

    def scrape(self):
        for buoycam_id in self.buoycam_ids:
            self._scrape_single_buoycam(buoycam_id)

    def _scrape_single_buoycam(self, buoycam_id):
        url = f"https://www.ndbc.noaa.gov/buoycam.php?station={buoycam_id}"
        response = requests.get(url)
        if response.status_code == 200:
            self._save_image(response.content, buoycam_id)
        else:
            print(f"Failed to retrieve image from buoycam {buoycam_id}")

    def _save_image(self, image_content, buoycam_id):
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"{buoycam_id}_{timestamp}.jpg"
        image_path = os.path.join(self.image_directory, buoycam_id, filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(image_content)
        print(f"Image saved: {image_path}")

class ImageProcessor:
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.panel_directory = os.path.join(base_directory, 'panels')
        os.makedirs(self.panel_directory, exist_ok=True)
        self.latest_images = {}  # Dictionary to hold the latest image per buoy

    def process_images(self):
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
                self._process_single_image(image)

    def _is_valid_image(self, image, threshold=10):
        return np.mean(image) >= threshold

    def _process_single_image(self, image):
        # Implement the processing logic such as enhancement, leveling, and so on
        pass

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