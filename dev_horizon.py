import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from horizon import process_image


class PanelProcessor:
    def __init__(self, panel):
        self.panel = panel
        self.gray_panel = cv2.cvtColor(self.panel, cv2.COLOR_BGR2GRAY)  # Convert to grayscale here
        self.edge_image = None
        self.rotated_panel = None

    #^ From GitHub Repo
    def detect_horizon_line(self):
        # Apply the detection logic from the provided script
        image_blurred = cv2.GaussianBlur(self.gray_panel, ksize=(3, 3), sigmaX=0)
        _, image_thresholded = cv2.threshold(
            image_blurred, thresh=0, maxval=1, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        image_thresholded = image_thresholded - 1
        image_closed = cv2.morphologyEx(image_thresholded, cv2.MORPH_CLOSE,
                                        kernel=np.ones((9, 9), np.uint8))
        horizon_x1 = 0
        horizon_x2 = self.gray_panel.shape[1] - 1
        horizon_y1 = max(np.where(image_closed[:, horizon_x1] == 0)[0])
        horizon_y2 = max(np.where(image_closed[:, horizon_x2] == 0)[0])
        return horizon_x1, horizon_x2, horizon_y1, horizon_y2

    def process(self):
        # Additional processing steps can be added here
        horizon_x1, horizon_x2, horizon_y1, horizon_y2 = self.detect_horizon_line()
        # Rotate the panel
        (h, w) = self.panel.shape[:2]
        center = (w // 2, h // 2)
        angle = np.rad2deg(np.arctan((horizon_y2 - horizon_y1) / (horizon_x2 - horizon_x1)))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.rotated_panel = cv2.warpAffine(self.panel, M, (w, h))
        # Center the horizon line
        # Code to move the horizon to the center on y-axis if necessary
        # ...
        return self.rotated_panel



    def save_panel(self, path):
        cv2.imwrite(path, self.rotated_panel)


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.panels = []
        self.modified_panels = []

    def crop_bottom_pixels(self, pixels=30):
        self.image = self.image[:-pixels, :]

    def split_into_panels(self):
        panel_width = self.image.shape[1] // 6
        self.panels = [self.image[:, i * panel_width:(i + 1) * panel_width] for i in range(6)]

    def process_panels(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, panel in enumerate(self.panels):

            # process the panel image
            panel_processed = process_image(panel)
            # save the processed panel image
            cv2.imwrite(os.path.join(output_dir, f'processed_panel_{i}.jpg'), panel_processed)
            # save the processed panel image with the horizon line
            plt.imshow(cv2.cvtColor(panel_processed, cv2.COLOR_BGR2GRAY), cmap='gray')
            plt.title(f'Processed Panel {i}')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'processed_panel_{i}_horizon.png'), bbox_inches='tight')
            plt.close()

            panel_processor = PanelProcessor(panel)
            horizon_x1, horizon_x2, horizon_y1, horizon_y2 = panel_processor.detect_horizon_line()
            rotated_panel = panel_processor.process()
            self.modified_panels.append(rotated_panel)
            # be sure to save the panels in the modified_panels directory
            filepath = os.path.join(output_dir, f'modified_panel_{i}.jpg')
            cv2.imwrite(filepath, rotated_panel)
            plt.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY), cmap='gray')
            plt.plot((horizon_x1, horizon_x2), (horizon_y1, horizon_y2), color='r', linewidth=2)
            plt.title(f'Panel {i} with Detected Horizon Line')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'panel_{i}_horizon.png'), bbox_inches='tight')
            plt.close()

    def stitch_panels(self, save_folder='modified_panels'):
        panorama = np.concatenate(self.modified_panels, axis=1)
        cv2.imwrite(f'{save_folder}/{os.path.basename(self.image_path)}', panorama)

    def execute(self):
        self.crop_bottom_pixels()
        self.split_into_panels()
        self.process_panels('processed_panels')

    def process_image(self, image_path, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        panel_processor = PanelProcessor(image)
        horizon_x1, horizon_x2, horizon_y1, horizon_y2 = panel_processor.detect_horizon_line()

        # Visualization
        plt.imshow(gray_image, cmap='gray')
        plt.plot((horizon_x1, horizon_x2), (horizon_y1, horizon_y2), color='r', linewidth=2)
        plt.title('Detected Horizon Line')
        plt.axis('off')

        plt.savefig(os.path.join(output_dir, 'horizon_line.png'), bbox_inches='tight')
        plt.close()

# # Usage
# test_image_path = 'images/buoys/42002/42002_20231218210342.jpg'
# save_folder = 'modified_panels'
# image_processor = ImageProcessor(test_image_path)
# image_processor.execute()
# Example usage:
output_dir = 'modified_panels'
image_processor = ImageProcessor('images/buoys/41009/41009_20231219181439.jpg')
image_processor.execute()