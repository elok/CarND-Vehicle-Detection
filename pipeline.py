"""
The goals / steps of this project are the following:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a
classifier Linear SVM classifier
- Optionally, you can also apply a color transform and append binned color features, as well as histograms of color,
to your HOG feature vector.
- Note: for those first two steps don't forget to normalize your features and randomize a selection for training and
testing.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4)
and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def classifier():
    """
    Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a
    classifier Linear SVM classifier
    :return:
    """
    pass


if __name__ == '__main__':
    classifier()

    # image = mpimg.imread('bbox-example-image.jpg')

    # Here are the bounding boxes I used
    # bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)),
    #           ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]
    #
    # result = draw_boxes(image, bboxes)
    # plt.imshow(result)