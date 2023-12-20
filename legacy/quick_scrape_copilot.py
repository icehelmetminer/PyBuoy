import datetime
import requests
import os
import re

def scrape_noaa_buoycams(image_directory):
    # URL of the buoycam image should be like this https://www.ndbc.noaa.gov/buoycam.php?station=42039
    buoycam_url = "https://www.ndbc.noaa.gov/buoycam.php?station={buoycam_id}"

    # List of buoycam IDs
    buoycam_ids = ["45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084"]
    # Create the image directory if it doesn't exist
    os.makedirs(image_directory, exist_ok=True)

    # Scrape images from each buoycam
    for buoycam_id in buoycam_ids:
        # Construct the URL for the buoycam image
        url = buoycam_url.format(buoycam_id=buoycam_id)

        # Send a GET request to retrieve the image data
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            timedateofimage = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            # Convert the timedateofimage to Zulu snake case format
            zulu_snakecased_time = re.sub(r'[^a-zA-Z0-9]', '_', timedateofimage)

            # Save the image to the image directory
            # Save the image with the Zulu snakecased timecode
            image_path = os.path.join(image_directory, f"{buoycam_id}/{buoycam_id}_{zulu_snakecased_time}.jpg")

            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"Image saved: {image_path}")
        else:
            print(f"Failed to retrieve image from buoycam {buoycam_id}")

# Example usage
image_directory = "images/buoys"

scrape_noaa_buoycams(image_directory)