## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in line 12 of file lesson_functions.py. The function is called get_hog_features().   

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

<img src="./output_images/car_hog_visualization_0.jpg" width="50%" height="50%">
<img src="./output_images/car_hog_visualization_1.jpg" width="50%" height="50%">
<img src="./output_images/car_hog_visualization_2.jpg" width="50%" height="50%">
<img src="./output_images/car_hog_visualization_3.jpg" width="50%" height="50%">
<img src="./output_images/car_hog_visualization_4.jpg" width="50%" height="50%">

<img src="./output_images/not_car_hog_visualization_0.jpg" width="50%" height="50%">
<img src="./output_images/not_car_hog_visualization_1.jpg" width="50%" height="50%">
<img src="./output_images/not_car_hog_visualization_2.jpg" width="50%" height="50%">
<img src="./output_images/not_car_hog_visualization_3.jpg" width="50%" height="50%">
<img src="./output_images/not_car_hog_visualization_4.jpg" width="50%" height="50%">

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and experimented with the results. Once, I have a good feel of what parameters provide me with good results, I like to go on slack and the discussion forums to see what others have come up with. A combination of the two gave me the best results.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in file model.py inside a function called train_and_return_svc(). Given all the necessary parameters, I do the following:

1. Read all the car and non-car images.
2. I pass the image into a function called extract_features() which first converts the image to the specified color space (YUV). Then it resizes the image to compute the binned color features. Then it calls color_hist() which generates a histogram out of all three channels of the image. And lastly, it calls the function get_hog_features() which returns the hog features of all three channels as well.
3. The feature set for both the car and non-car are then stacked and normalize using sklearn's StandardScaler.
4. Once the features are normalized, we label the cars with 1's and non-cars with 0's. 
5. The data is split up into randomized training and test sets.
6. The training set is then passed into the sklearn's LinearSVC classifier.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My code for sliding window search is located in file model.py under function process_image() (code line starting at 212). I took the advice of the lessons to not search the sky and to search certain scales in only certain areas of the image -- so small scales around the center of the image and larger scales towards the bottom of the image. This took alot of trial and error. I had to output alot of visuals to help debug where I was actually searching and if it made any sense.

<img src="./output_images/window_search_1.jpg">
<img src="./output_images/window_search_2.jpg">
<img src="./output_images/window_search_3.jpg">
<img src="./output_images/window_search_4.jpg">
<img src="./output_images/window_search_5.jpg">
<img src="./output_images/window_search_6.jpg">

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 6 scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result:

ystart = 400
ystop = 600
scale = 0.5

ystart = 400
ystop = 600
scale = 1.0

ystart = 400
ystop = 600
scale = 1.3

ystart = 400
ystop = 600
scale = 1.5

ystart = 400
ystop = 720
scale = 2

ystart = 400
ystop = 820
scale = 2.5

Here are some example images:

<img src="./output_images/pipeline_result_1.jpg">
<img src="./output_images/pipeline_result_2.jpg">
<img src="./output_images/pipeline_result_3.jpg">
<img src="./output_images/pipeline_result_4.jpg">

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I saved down the 15 most recent bounding boxes and created a heatmap and then thresholded that map by 2.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The code to save the 15 most recent bounding boxes is located in model.py add_bbox() (line 53).

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

<img src="./output_images/heatmap_example_1.jpg">
<img src="./output_images/heatmap_example_2.jpg">
<img src="./output_images/heatmap_example_3.jpg">
<img src="./output_images/heatmap_example_4.jpg">
<img src="./output_images/heatmap_example_5.jpg">
<img src="./output_images/heatmap_example_6.jpg">

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Hyperparameter Tuning - tuning the parameters were mostly through trial and error. I would like to explore using GridSearchCV and RandomizedSearchCV. 

Debugging - it was tough at times to see what was going on with my heatmaps and window searches. I was able to add an inset/overlay on top of my output and that helped tremendously to debug my issues.

Caching / False Positives - I still have false positives. I'm not really sure the best way to fix this for good. Some thoughts are tweaking the amount of previous bounding boxes to cache. Another thought is some way to average the heat maps across frames. Maybe one way is to use deep learning instead of SVC to classify a car vs non-car. I'm curious to see if that would improve the results.

Integrate Advance Lane Finding - I'd love to integrate project 4 with project 5 and be able to see both the lane lines highlighted and cars on the road identified.
