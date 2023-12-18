import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
import time
import datetime
import requests
import os
import re
import cv2
import numpy as np
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
import time
import datetime
import requests
import os
import re
import cv2
import numpy as np
os.makedirs('panels', exist_ok=True)
panel_ids = glob.glob('images/*/*')
panels = glob.glob('panels/*')
def scrape_noaa_buoycams(image_directory):
    buoycam_url = "https://www.ndbc.noaa.gov/buoycam.php?station={buoycam_id}"
    buoycam_ids = ["45007", "45012", "46002", "46011", "46012", "46015", "46025", "46026", "46027", "46028", "46042", "46047", "46053", "46054", "46059", "46066", "46069", "46071", "46072", "46078", "46085", "46086", "46087", "46088", "46089", "51000", "51001", "51002", "51003", "51004", "51101", "46084"]
    os.makedirs(image_directory, exist_ok=True)
    for buoycam_id in buoycam_ids:
        url = buoycam_url.format(buoycam_id=buoycam_id)
        response = requests.get(url)
        if response.status_code == 200:
            timedateofimage = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            zulu_snakecased_time = re.sub(r'[^a-zA-Z0-9]', '_', timedateofimage)
            image_path = os.path.join(image_directory, f"{buoycam_id}/{buoycam_id}_{zulu_snakecased_time}.jpg")
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"Image saved: {image_path}")
        else:
            print(f"Failed to retrieve image from buoycam {buoycam_id}")
image_directory = "images/buoys"
scrape_noaa_buoycams(image_directory)
def is_mostly_black(image, threshold=10):
    return np.mean(image) < threshold
def stitch_panels_horizontally(panels):
    max_height = max(panel.shape[0] for panel in panels)
    panels_resized = [cv2.resize(panel, (panel.shape[1], max_height), interpolation=cv2.INTER_LINEAR) for panel in panels]
    return np.concatenate(panels_resized, axis=1)
def stitch_vertical(rows):
    max_width = max(row.shape[1] for row in rows)
    rows_resized = []
    for row in rows:
        if row.shape[1] < max_width:
            padding = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
            row_resized = np.concatenate((row, padding), axis=1)
        else:
            row_resized = row
        rows_resized.append(row_resized)
    return np.concatenate(rows_resized, axis=0)
def split_into_panels(image, number_of_panels=6):
    width = image.shape[1]
    panel_width = width // number_of_panels
    panels = [image[:, i*panel_width:(i+1)*panel_width] for i in range(number_of_panels)]
    panels[-1] = image[:, (number_of_panels-1)*panel_width:]
    return panels
def remove_bottom_strip(panel, strip_height=20):
    return panel[:-strip_height, :]
def enhance_image(panel, clip_limit=2.0, tile_grid_size=(8, 8)):
    panel_yuv = cv2.cvtColor(panel, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    panel_yuv[:, :, 0] = clahe.apply(panel_yuv[:, :, 0])
    enhanced_panel = cv2.cvtColor(panel_yuv, cv2.COLOR_YUV2BGR)
    return enhanced_panel
def preprocess_and_stitch_panels(image, number_of_panels=6, strip_height=35):
    panels = split_into_panels(image, number_of_panels)
    processed_panels = []
    for panel in panels:
        panel = enhance_image(panel)
        processed_panels.append(panel)
        temp_panel = panel.copy()
        cv2.imshow('panel', panel)
        cv2.waitKey(10)
    return stitch_panels_horizontally(processed_panels)
def check_for_duplicate_panel(image):
    for panel in panels:
        panel = cv2.imread(panel)
        if np.array_equal(image, panel):
            return True
    return False
files = glob.glob('images/buoys/*/*')
rows_to_stitch = []
latest_image_files = []
for file in tqdm(files):
    image = cv2.imread(file)
    orange_value = np.mean(image[:,:,2])
    if not (10 <= orange_value <= 150) or is_mostly_black(image):
        continue
    elif latest_image_files and np.array_equal(image, cv2.imread(latest_image_files[-1])):
        continue
    row = preprocess_and_stitch_panels(image)
    rows_to_stitch.append(row)
    latest_image_files.append(file)
collage = stitch_vertical(rows_to_stitch)
def check_for_duplicate_panel(image):
    for panel in panels:
        panel = cv2.imread(panel)
        if np.array_equal(image, panel):
            print('found duplicate panel')
            return True
    return False
for row in rows_to_stitch:
    if check_for_duplicate_panel(row):
        rows_to_stitch.remove(row)
        print('removed duplicate row')
timestamp = str(int(time.time()))
collage_directory = 'images/save_images'
os.makedirs(collage_directory, exist_ok=True)
filename = os.path.join(collage_directory, f'collage_{timestamp}.jpg')
cv2.imwrite(filename, collage)
print(f"Collage saved to {filename}")
cv2.imshow('Vertical Collage', collage)
cv2.waitKey(0)
cv2.destroyAllWindows()
for row in rows_to_stitch:
    if check_for_duplicate_panel(row):
        rows_to_stitch.remove(row)
        print('removed duplicate row')
timestamp = str(int(time.time()))
collage_directory = 'images/save_images'
os.makedirs(collage_directory, exist_ok=True)
filename = os.path.join(collage_directory, f'collage_{timestamp}.jpg')
cv2.imwrite(filename, collage)
print(f"Collage saved to {filename}")
cv2.imshow('Vertical Collage', collage)
cv2.waitKey(0)
cv2.destroyAllWindows()
