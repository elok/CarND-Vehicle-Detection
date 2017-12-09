import numpy as np

class HeatMap():

    def __init__(self, img):
        self.heat_map = np.zeros_like(img[:, :, 0]).astype(np.float)
        # self.cold_increment = 1
        self.hot_increment = 1
        self.bounding_boxes = []
        self.threshold = 1

    def add_heat(self, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += self.hot_increment

        # Return updated heatmap
        return self.heat_map

    def apply_threshold(self):
        # Zero out pixels below the threshold
        self.heat_map[self.heat_map <= self.threshold] = 0
        # Return thresholded map
        return self.heat_map

    # def chill(self):
    #     self.heat_map -= self.cold_increment