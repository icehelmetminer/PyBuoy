import cv2
import numpy as np
import os

class PanelProcessor:
    def __init__(self, panel):
        self.panel = panel
        self.gray_panel = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)  # Convert to grayscale here
        self.rotated_panel = None


    def increase_contrast(self):
        # Now we're sure that self.gray_panel is a grayscale image
        return cv2.equalizeHist(self.gray_panel)

    def detect_edges(self):
        high_contrast = self.increase_contrast()
        # Now we're sure that high_contrast is a grayscale image
        # show it for 4 seconds
        # cv2.imshow('high_contrast', high_contrast)
        # cv2.waitKey(1000)
        return cv2.Canny(high_contrast, 50, 150)

    def find_rotation_angle(self):
        # Assuming detect_edges has been called
        lines = cv2.HoughLines(self.edge_image, 1, np.pi / 180, 150)
        if lines is not None:
            # find longest line that has the highest contrast on either side
            h_line = None
            h_line_length = 0
            for line in lines:
                for rho, theta in line:
                    # Find the longest line near the horizontal orientation
                    if abs(np.pi / 2 - theta) < np.pi / 6:
                        # Find the length of the line
                        length = np.sqrt(self.panel.shape[0] ** 2 + self.panel.shape[1] ** 2)
                        if length > h_line_length:
                            h_line = line
                            h_line_length = length
            if h_line is not None:
                rho, theta = h_line[0]
                angle = (theta - np.pi / 2) * 180 / np.pi
                return -angle
        else:
            return 0

    def rotate_and_center_horizon(self):
        angle = self.find_rotation_angle()
        (h, w) = self.panel.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.rotated_panel = cv2.warpAffine(self.panel, M, (w, h))
        # Center the horizon line
        # Code to move the horizon to the center on y-axis if necessary
        # ...
        return self.rotated_panel

    def process(self):
        self.edge_image = self.detect_edges()
        self.rotated_panel = self.rotate_and_center_horizon()
        return self.rotated_panel


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

    def process_panels(self):
        for i, panel in enumerate(self.panels):
            panel_processor = PanelProcessor(panel)
            rotated_panel = panel_processor.process()
            self.modified_panels.append(rotated_panel)
            # be sure to save the panels in the modified_panels directory
            filepath = os.path.join('modified_panels', f'modified_panel_{i}.jpg')
            cv2.imwrite(filepath, rotated_panel)

    def stitch_panels(self):
        panorama = np.concatenate(self.modified_panels, axis=1)
        cv2.imwrite('modified_panels/panorama.jpg', panorama)

    def execute(self):
        self.crop_bottom_pixels()
        self.split_into_panels()
        self.process_panels()
        self.stitch_panels()

# Usage
image_processor = ImageProcessor('images/buoys/42002/42002_20231218210342.jpg')
image_processor.execute()
