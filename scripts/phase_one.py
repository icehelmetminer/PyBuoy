import requests
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from datetime import datetime
from ratelimit import limits, sleep_and_retry
import matplotlib.pyplot as plt
import pandas as pd

collecting_all = True
buoy_list = pd.read_csv("scripts/working_buoys.csv")

def get_image_size(img):
    return img.size

def mse_between_arrays(arr1, arr2):
    try:
        return np.mean((arr1 - arr2) ** 2)
    except:
        return 0

def crop_the_bottom_off(images):
    if not isinstance(images, list):
        images = [images]

    for image in images:
        try:
            img_width, img_height = get_image_size(image)
            cropped_image = image.crop((0, 0, img_width, img_height-20))
            cropped_image.save(image)
        except Exception as e:
            print("Error cropping the bottom off of the image: " + str(e))

@sleep_and_retry
@limits(calls=15, period=60)
def download_image(image_url):
    global buoy_list
    response = requests.get(image_url)
    if response.status_code != 200:
        print(f"API response: {response.status_code}")
    img = Image.open(BytesIO(response.content))
    img_array = np.asarray(img)

    if np.sum(img_array > 200) / img_array.size > 0.9:
        print(f"Image too white: {image_url}")

        with open("scripts/failing_buoys.csv", "r") as f:
            failing_buoys = f.read().splitlines()
        if image_url not in failing_buoys:
            with open("scripts/failing_buoys.csv", "a") as f:
                f.write(image_url + "\n")

        buoy_id = image_url.split("/")[-1].split(".")[0]

        with open("scripts/failing_buoys.csv", "a") as f:
            f.write(buoy_id + "\n")

        buoy_list.to_csv("scripts/working_buoys.csv", index=False)
        return None
    return img

def resize_image_to_standard_height(image, target_height):
    if image is None:
        return None
    width, height = image.size
    new_height = target_height
    new_width = int((new_height / height) * width)
    return image.resize((new_width, new_height), Image.ANTIALIAS)

def split_image_into_panels(resized_image, num_panels):
    if resized_image is None:
        return None
    width, height = resized_image.size
    panel_width = width // num_panels

    panels = []
    for i in range(num_panels):
        left = i * panel_width
        right = left + panel_width
        panel = resized_image.crop((left, 0, right, height))
        panel = panel.crop((0, 0, panel_width, height - 38))
        panels.append(panel)

    return panels

def check_unusual_panels(panels, mse_threshold):
    unusual_panels = []
    rich_color_panels = []
    panel_mses = []
    for panel in panels:
        panel_array = np.array(panel)
        panel_mse = mse_between_arrays(panel_array, panel_array)
        panel_mses.append(panel_mse)

    median_mse = np.median(panel_mses)
    rich_color_threshold = median_mse * 0.8

    for i, panel in enumerate(panels):
        panel_array = np.array(panel)
        panel_mse = panel_mses[i]
        if panel_mse > mse_threshold:
            unusual_panels.append(i)
        if panel_mse > rich_color_threshold:
            rich_color_panels.append(panel)
    return unusual_panels, rich_color_panels

def detect_horizon_line(img):
    img = np.array(img)

    if np.sum(img > 200) / img.size < 0.9:
        plt.savefig('latest_horizon_line.png')
    else:
        pass

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    try:
        longest_line = max(lines, key=lambda line: np.linalg.norm(line[0][2:] - line[0][:2]))
        x1, y1, x2, y2 = longest_line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle
    except TypeError:
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

def align_horizon_line(img):
    img = np.array(img)
    original = img.copy()
    tilt_angle = detect_horizon_line(img)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, tilt_angle, 1)
    aligned_img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    fixed = aligned_img.copy()

    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[1].imshow(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Aligned {}".format(tilt_angle))
    plt.savefig('latest_alignment.png')

    return aligned_img

def check_unusual_panels(panels, mse_threshold):
    unusual_panels = []
    rich_color_panels = []
    panel_mses = []
    for panel in panels:
        panel_array = np.array(panel)
        panel_mse = mse_between_arrays(panel_array, panel_array)
        panel_mses.append(panel_mse)

    median_mse = np.median(panel_mses)
    rich_color_threshold = median_mse * 0.8

    for i, panel in enumerate(panels):
        panel_array = np.array(panel)
        panel_mse = panel_mses[i]
        if panel_mse > mse_threshold:
            unusual_panels.append(i)
        if panel_mse > rich_color_threshold:
            rich_color_panels.append(panel)
    return unusual_panels, rich_color_panels

def detect_horizon_line(img):
    img = np.array(img)

    if np.sum(img > 200) / img.size < 0.9:
        plt.savefig('latest_horizon_line.png')
    else:
        pass

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    try:
        longest_line = max(lines, key=lambda line: np.linalg.norm(line[0][2:] - line[0][:2]))
        x1, y1, x2, y2 = longest_line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle
    except TypeError:
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

def align_horizon_line(img):
    img = np.array(img)
    original = img.copy()
    tilt_angle = detect_horizon_line(img)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, tilt_angle, 1)
    aligned_img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    fixed = aligned_img.copy()

    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[1].imshow(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Aligned {}".format(tilt_angle))
    plt.savefig('latest_alignment.png')

    return aligned_img

def stitch_aligned_images(aligned_images):
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    if len(aligned_images) < 2:
        print("Not enough images to stitch")
        return None
    orb = cv2.ORB_create(nfeatures=1000)

    for i, img in enumerate(aligned_images):
        keypoints, _ = orb.detectAndCompute(img, None)
        print(f"Image {i + 1} has {len(keypoints)} features")

    (status, stitched_image) = stitcher.stitch(aligned_images)

    if status == 0:
        return stitched_image
    else:
        return None

def main():

    buoy_list_df = pd.read_csv("scripts/working_buoys.csv")

    buoy_list = buoy_list_df["station_id"].tolist()

    buoy_urls = [f'https://www.ndbc.noaa.gov/buoycam.php?station={station}' for station in buoy_list]

    buoy_list_df["buoy_url"] = buoy_urls

    base_output_path = "images/output_test"

    print(f'There are {len(buoy_urls)} buoys to process')

    for buoy_url in tqdm(buoy_urls, desc="Processing buoys"):
        print(f'Processing {buoy_url}', end='', flush=True)
        image = download_image(buoy_url)
        if image is None:
            continue

        resized_image = resize_image_to_standard_height(image, target_height=640)
        panels = split_image_into_panels(resized_image, num_panels=4)
        unusual_panels, rich_color_panels = check_unusual_panels(panels, mse_threshold=1)

        if unusual_panels:
            print(f" - Unusual panels: {unusual_panels}")
            current_datetime = datetime.now()
            date_str = current_datetime.strftime("%Y-%m-%d")
            time_str = current_datetime.strftime("%H-%M-%S")
            buoy_name = buoy_url.split("/")[-2]

            output_path = os.path.join(base_output_path, buoy_name, date_str, time_str)
            os.makedirs(output_path, exist_ok=True)

            for i, panel in enumerate(panels):
                panel_output_path = os.path.join(output_path, f"panel_{i+1}.jpg")
                panel.save(panel_output_path)

            aligned_images = [align_horizon_line(panel) for panel in panels]
            stitched_image = stitch_aligned_images(aligned_images)

            panorama_output_path = os.path.join(output_path, "panorama.jpg")
            try:
                cv2.imwrite(panorama_output_path, stitched_image)

                latest_output_path = os.path.join(base_output_path, buoy_name, "latest.jpg")
                cv2.imwrite(latest_output_path, stitched_image)
            except AttributeError:
                print("Panorama could not be stitched")
            except Exception as e:
                pass

        else:
            print(" - No unusual panels")

if __name__ == "__main__":
    main()