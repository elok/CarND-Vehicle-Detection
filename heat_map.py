import numpy as np

class HeatMap():

    def __init__(self, img):
        self.heat_map = np.zeros_like(img[:, :, 0]).astype(np.float)
        self.hot_increment = 1
        self.bounding_boxes = []
        self.heat_map_rgb = np.zeros_like(img).astype(np.float)
        # self.threshold = 1

    def add_heat(self, bbox_list):

        # chill the whole heat map by one
        self.heat_map -= 1
        self.heat_map = np.clip(self.heat_map, 0, 255)

        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            if box:
                self.heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += self.hot_increment

        # Return updated heatmap
        return self.heat_map

    def apply_threshold(self, threshold):
        # Zero out pixels below the threshold
        self.heat_map[self.heat_map <= threshold] = 0
        # Return thresholded map
        return self.heat_map

    def get_heat_map_as_RGB(self):
        self.heat_map_rgb[:, :, 0] = self.heat_map
        self.heat_map_rgb[:, :, 1] = self.heat_map
        self.heat_map_rgb[:, :, 2] = self.heat_map
        return self.heat_map_rgb