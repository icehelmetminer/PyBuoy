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
# from skimage.transform import rotate

# keep the panels in a folder called panels in the same directory as the images or in the /images folder
# make it if it doesn't exist
os.makedirs('panels', exist_ok=True)
# make a list of all the images in the images folder
panel_ids = glob.glob('images/*/*')
# make a list of all the panels in the panels folder
panels = glob.glob('panels/*')



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
# Function to determine if an image is mostly black
def is_mostly_black(image, threshold=10):
    return np.mean(image) < threshold

# Function to stitch images horizontally
def stitch_panels_horizontally(panels):
    # Ensure all panels are the same height before stitching
    max_height = max(panel.shape[0] for panel in panels)
    panels_resized = [cv2.resize(panel, (panel.shape[1], max_height), interpolation=cv2.INTER_LINEAR) for panel in panels]
    return np.concatenate(panels_resized, axis=1)

# Function to stitch images vertically
def stitch_vertical(rows):
    # Ensure all rows are the same width before stitching
    max_width = max(row.shape[1] for row in rows)
    # Resize rows to the max width or pad with black pixels
    rows_resized = []
    for row in rows:
        if row.shape[1] < max_width:
            padding = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
            row_resized = np.concatenate((row, padding), axis=1)
        else:
            row_resized = row
        rows_resized.append(row_resized)

    # Stitch the rows together
    return np.concatenate(rows_resized, axis=0)
# Split into vertical panels
def split_into_panels(image, number_of_panels=6):
    # Split the image into six equal vertical panels (side by side)
    width = image.shape[1]
    panel_width = width // number_of_panels
    panels = [image[:, i*panel_width:(i+1)*panel_width] for i in range(number_of_panels)]
    # Ensure last panel takes any remaining pixels to account for rounding
    panels[-1] = image[:, (number_of_panels-1)*panel_width:]
    return panels

# Remove the bottom strip from each panel
def remove_bottom_strip(panel, strip_height=20):
    # Assume the strip to be removed is at the bottom of the image
    return panel[:-strip_height, :]

# Enhance the image
def enhance_image(panel, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Apply any enhancements to the panel, like histogram equalization, etc.
    # Convert to YUV color space
    panel_yuv = cv2.cvtColor(panel, cv2.COLOR_BGR2YUV)
    # Apply CLAHE to the Y channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    panel_yuv[:, :, 0] = clahe.apply(panel_yuv[:, :, 0])
    # Convert back to BGR color space
    enhanced_panel = cv2.cvtColor(panel_yuv, cv2.COLOR_YUV2BGR)
    return enhanced_panel

# # Process and stitch panels
# def preprocess_and_stitch_panels(image, number_of_panels=6, strip_height=35):
#     panels = split_into_panels(image, number_of_panels)
#     processed_panels = [enhance_image(remove_bottom_strip(panel, strip_height)) for panel in panels]
#     return stitch_panels_horizontally(processed_panels)
# #* New Functions
# def find_horizon_line(image):
#     try:
#         # Apply edge detection
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150)

#         # Use Hough transform to find lines
#         lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

#         # Assume the horizon is the longest line
#         horizon_line = sorted(lines, key=lambda x: np.linalg.norm((x[0][0] - x[0][2], x[0][1] - x[0][3])), reverse=True)[0][0]
#         return horizon_line
#     except Exception as e:
#         print(e)
#         return None

# def rotate_image(image, angle):
#     # Compute the rotation matrix
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)

#     # Perform the rotation
#     rotated = cv2.warpAffine(image, M, (w, h))
#     return rotated

# def align_horizon(image):
#     line = find_horizon_line(image)
#     if line is None:
#         return image  # No line found, return original image

#     # Calculate angle to horizontal
#     angle = np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0]))

#     # Rotate the image to align the horizon
#     aligned_image = rotate_image(image, -angle)

#     # draw a red line on the image at the horizon
#     # cv2.line(aligned_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

#     #! show the image temporarily to check if the horizon is aligned (goes by automatically)
#     # flash the image on the screen for 1 second
#     # cv2.imshow('aligned_image', aligned_image)
#     cv2.waitKey(1000)
#     cv2.destroyAllWindows()
#     # cv2.waitKey(0)

#     return aligned_image

# # Function to crop the images to the same height based on the horizon line
# def crop_to_horizon(images):
#     # Find the horizon line for each image and store the y-coordinate
#     horizons_y = [find_horizon_line(image)[1] for image in images]

#     # Determine the maximum y-coordinate, which corresponds to the lowest horizon line
#     max_y = max(horizons_y)

#     # Crop each image to this horizon line
#     cropped_images = [image[max_y:image.shape[0] - max_y, :] for image in images]
#     return cropped_images

# # New function to level the horizon
# def rotate_to_level_horizon(panel, line):
#     if line is None:
#         return panel  # Return original if no line found

#     x1, y1, x2, y2 = line
#     angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
#     rotated_panel = cv2.warpAffine(panel, cv2.getRotationMatrix2D((panel.shape[1]//2, panel.shape[0]//2), angle, 1.0), (panel.shape[1], panel.shape[0]))
#     return rotated_panel

# #^modified
# # Modify preprocess_and_stitch_panels function
def preprocess_and_stitch_panels(image, number_of_panels=6, strip_height=35):
    panels = split_into_panels(image, number_of_panels)
    processed_panels = [] # Store the processed panels
    for panel in panels:
        #& panel = remove_bottom_strip(panel, strip_height)
        panel = enhance_image(panel)
        processed_panels.append(panel)
        temp_panel = panel.copy()
        # annotate this temp_panel with orange value
        # orange_value = np.mean(temp_panel[:,:,2])
        # # then if the image is mostly black put red text on the image that says "night" and show it
        # if not (10 <= orange_value <= 150) or is_mostly_black(temp_panel):
        #     cv2.putText(temp_panel, 'night', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # # otherwise put the orange value on the image and show it
        # else:
        #     cv2.putText(temp_panel, str(orange_value), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #     # show the temp_panel using cv2.imshow
        # cv2.imshow('temp_panel', temp_panel)
        # wait for a moment
        # cv2.waitKey(10)
        # show the panel using cv2.imshow (this is the original panel)
        cv2.imshow('panel', panel)
        # wait for a moment
        cv2.waitKey(10)
    return stitch_panels_horizontally(processed_panels)

# def rotate_image(image, angle):
#     # Grab the dimensions of the image and then determine the center

#     # only rotate the image if it has not been rotated yet
#     if angle == 0:
#         print(f'Found a panel that has not been rotated yet. Rotating {angle} degrees')
#         return image

#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)

#     # Grab the rotation matrix, then grab the sine and cosine
#     M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])

#     # Compute the new bounding dimensions of the image
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))

#     # Adjust the rotation matrix to take into account translation
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY

#     # Perform the actual rotation and return the image
#     return cv2.warpAffine(image, M, (nW, nH))

# def align_and_crop_panels(panels):
#     horizon_levels = []
#     for panel in panels:
#         horizon_line = find_horizon_line(panel)
#         if horizon_line is not None:
#             angle = np.arctan2(horizon_line[3] - horizon_line[1], horizon_line[2] - horizon_line[0]) * (180 / np.pi)
#             rotated_panel = rotate_image(panel, -angle)
#             panels[panels.index(panel)] = rotated_panel
#             # Store the y-coordinate of the horizon
#             horizon_levels.append(horizon_line[1] if angle > 0 else horizon_line[3])
#         else:
#             horizon_levels.append(panel.shape[0] // 2)  # Assume horizon is in the middle if not detected

#     # Determine the lowest horizon level across all panels
#     max_horizon_level = max(horizon_levels)

#     # Crop panels to align the horizons and to the same height
#     cropped_panels = []
#     for panel in panels:
#         # Calculate cropping indices
#         y_offset = max_horizon_level - (panel.shape[0] // 2)
#         crop_top = max(y_offset, 0)
#         crop_bottom = panel.shape[0] - max(y_offset + panel.shape[0] // 2, 0)
#         cropped_panel = panel[crop_top:crop_bottom, :]
#         cropped_panels.append(cropped_panel)

#     # Stitch the cropped panels horizontally
#     stitched_row = stitch_panels_horizontally(cropped_panels)
#     return stitched_row

# #^ end modified


def check_for_duplicate_panel(image):
    # check the image against all the panels in the panels folder with and without enhancement or any rotation.
    # if it matches any of them, return True and do not save the image
    for panel in panels:
        panel = cv2.imread(panel)
        # panel = enhance_image(panel)
        # panel = rotate(panel, angle=180)
        if np.array_equal(image, panel):
            return True
    return False


# Main processing logic
files = glob.glob('images/buoys/*/*')
rows_to_stitch = []
latest_image_files = []  # Maintain a list of filenames that have been processed

for file in tqdm(files):
    image = cv2.imread(file)
    orange_value = np.mean(image[:,:,2])

    if not (10 <= orange_value <= 150) or is_mostly_black(image):
        continue
    # Compare with the last image file processed, not the processed row
    elif latest_image_files and np.array_equal(image, cv2.imread(latest_image_files[-1])):
        continue

    row = preprocess_and_stitch_panels(image)
    rows_to_stitch.append(row)  # Append the row for stitching
    latest_image_files.append(file)  # Append the file to the list for comparison


# Create a collage from the rows
collage = stitch_vertical(rows_to_stitch)


def check_for_duplicate_panel(image):
    # check the image against all the panels in the panels folder with and without enhancement or any rotation.
    # if it matches any of them, return True and do not save the image
    for panel in panels:
        panel = cv2.imread(panel)
        # panel = enhance_image(panel)
        # panel = rotate(panel, angle=180)
        if np.array_equal(image, panel):
            print('found duplicate panel')
            return True
    return False

# remove duplicate rows
for row in rows_to_stitch:
    if check_for_duplicate_panel(row):
        rows_to_stitch.remove(row)
        print('removed duplicate row')

# Save the collage with timestamp
timestamp = str(int(time.time()))
collage_directory = 'images/save_images'
os.makedirs(collage_directory, exist_ok=True)
filename = os.path.join(collage_directory, f'collage_{timestamp}.jpg')
cv2.imwrite(filename, collage)
print(f"Collage saved to {filename}")

# Display the collage
cv2.imshow('Vertical Collage', collage)
cv2.waitKey(0)
cv2.destroyAllWindows()



# remove duplicate rows
for row in rows_to_stitch:
    if check_for_duplicate_panel(row):
        rows_to_stitch.remove(row)
        print('removed duplicate row')


# Save the collage with timestamp
timestamp = str(int(time.time()))
collage_directory = 'images/save_images'
os.makedirs(collage_directory, exist_ok=True)
filename = os.path.join(collage_directory, f'collage_{timestamp}.jpg')
cv2.imwrite(filename, collage)
print(f"Collage saved to {filename}")

# Display the collage
cv2.imshow('Vertical Collage', collage)
cv2.waitKey(0)
cv2.destroyAllWindows()
