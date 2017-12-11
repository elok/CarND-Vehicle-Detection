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
import matplotlib.pyplot as plt
import pickle
import os
from moviepy.editor import VideoFileClip
from lesson_functions import *
import glob
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from heat_map import HeatMap

COLOR_SPACE = 'YUV'
CONVERT_COLOR_SPACE = 'BGR2YUV'

# Min and max in y to search in slide_window()
ystart = 400
ystop = 656
scale = 1.5

orient = 11  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
spatial_size = (32, 32)  # Spatial binning dimensions (aka resizing)
hist_bins = 32  # Number of histogram bins
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

class MasterVehicleDetection():

    def __init__(self, img, svc, X_scaler):
        self.svc = svc
        self.X_scaler = X_scaler
        self.heat_map = HeatMap(img)
        self.num_frame = 0
        self.last_heat_img = []
        self.last_img_with_all_searched_bound_box = None

    def debug_show_boxes(self, img, bbox_list, title='', frame=0):
        if bbox_list:
            img_w_car_bbox = draw_boxes(img, bbox_list)

            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            fig.subplots_adjust(hspace=0.4, wspace=0.05)

            axes[0].imshow(img_w_car_bbox)
            sub_title = "img_w_car_bbox"
            axes[0].set_title(sub_title, fontsize=11)

            axes[1].imshow(self.last_img_with_all_searched_bound_box)
            sub_title = "last_img_with_all_searched_bound_box"
            axes[1].set_title(sub_title, fontsize=11)

            axes[2].imshow(self.heat_map.heat_map)
            sub_title = "heat_map"
            axes[2].set_title(sub_title, fontsize=11)

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.suptitle(title)

            plt.show()
            # plt.savefig('heat_maps/heatmap_{0}.jpg'.format(frame))

    def find_cars(self, img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                  hist_bins, hog_channel):
        """
        Define a single function that can extract features using hog sub-sampling and make predictions
        :param img:
        :param ystart:
        :param ystop:
        :param scale:
        :param svc:
        :param X_scaler:
        :param orient:
        :param pix_per_cell:
        :param cell_per_block:
        :param spatial_size:
        :param hist_bins:
        :param hog_channel:
        :return:
        """
        draw_img = np.copy(img)
        # img = img.astype(np.float32) / 255

        # crop the image to the start and stop in the y axis
        img_tosearch = img[ystart:ystop, :, :]
        # color transform to search
        ctrans_tosearch = convert_color(img_tosearch, conv=CONVERT_COLOR_SPACE)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        # extract the 3 color channels
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        bbox_list = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                # Extract HOG for this patch
                if hog_channel == 'ALL':
                    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = get_hog_features(ctrans_tosearch[:, :, hog_channel], orient,
                                                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                # For debug purposes, draw a box around all the locations we're searching
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (255, 0, 0), 6)

                if test_prediction == 1:
                    # cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                    #               (xbox_left + win_draw, ytop_draw + win_draw + ystart), (255, 0, 0), 6)

                    bbox_list.append(
                        ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

        # For debug purposes, save the image with all the bounding box locations we're searching
        self.last_img_with_all_searched_bound_box = draw_img

        return bbox_list

    def process_image(self, img):
        bound_box_list_all = []
        debug_plot = DebugPlot(title='Frame: {0}'.format(self.num_frame))

        if len(self.last_heat_img) > 0:
            debug_plot.add_images(title='last_heat_img : frame: {0}'.format(self.num_frame),
                                 img_1=self.last_heat_img, subtitle_1='last_heat_img',
                                 img_2=self.last_heat_img, subtitle_2='last_heat_img', convert=False)

        ystart = 300
        ystop = 720
        scale = 1.0
        bbox_list_curr = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, orient, pix_per_cell,
                                        cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)
        if bbox_list_curr:
            bound_box_list_all.append(bbox_list_curr)

        img_w_car_bbox = draw_boxes(img, bbox_list_curr)
        debug_plot.add_images(title='Small cars on horizon - scale 1 : frame: {0}'.format(self.num_frame),
                              img_1=img_w_car_bbox, subtitle_1='img_w_car_bbox',
                              img_2=self.last_img_with_all_searched_bound_box, subtitle_2='img_w_all_bbox')

        ystart = 300
        ystop = 720
        scale = 1.5
        bbox_list_curr = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, orient, pix_per_cell,
                                   cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)
        if bbox_list_curr:
            bound_box_list_all.append(bbox_list_curr)

        img_w_car_bbox = draw_boxes(img, bbox_list_curr)
        debug_plot.add_images(title='Small cars on horizon - scale 1.5 : frame: {0}'.format(self.num_frame),
                              img_1=img_w_car_bbox, subtitle_1='img_w_car_bbox',
                              img_2=self.last_img_with_all_searched_bound_box, subtitle_2='img_w_all_bbox')


        ystart = 400
        ystop = 720
        scale = 1
        bbox_list_curr = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, orient, pix_per_cell,
                                   cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)
        if bbox_list_curr:
            bound_box_list_all.append(bbox_list_curr)

        img_w_car_bbox = draw_boxes(img, bbox_list_curr)
        debug_plot.add_images(title='medium cars on horizon - scale 1 : frame: {0}'.format(self.num_frame),
                              img_1=img_w_car_bbox, subtitle_1='img_w_car_bbox',
                              img_2=self.last_img_with_all_searched_bound_box, subtitle_2='img_w_all_bbox')


        ystart = 400
        ystop = 720
        scale = 1
        bbox_list_curr = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, orient, pix_per_cell,
                                   cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)
        if bbox_list_curr:
            bound_box_list_all.append(bbox_list_curr)

        img_w_car_bbox = draw_boxes(img, bbox_list_curr)
        debug_plot.add_images(title='large cars on horizon - scale 1.5 : frame: {0}'.format(self.num_frame),
                              img_1=img_w_car_bbox, subtitle_1='img_w_car_bbox',
                              img_2=self.last_img_with_all_searched_bound_box, subtitle_2='img_w_all_bbox')

        ystart = 400
        ystop = 720
        scale = 1.5
        bbox_list_curr = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, orient, pix_per_cell,
                                   cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)
        if bbox_list_curr:
            bound_box_list_all.append(bbox_list_curr)

        img_w_car_bbox = draw_boxes(img, bbox_list_curr)
        debug_plot.add_images(title='large cars on horizon - scale 1 : frame: {0}'.format(self.num_frame),
                              img_1=img_w_car_bbox, subtitle_1='img_w_car_bbox',
                              img_2=self.last_img_with_all_searched_bound_box, subtitle_2='img_w_all_bbox')

        ystart = 400
        ystop = 720
        scale = 2.0
        bbox_list_curr = self.find_cars(img, ystart, ystop, scale, self.svc, self.X_scaler, orient, pix_per_cell,
                                   cell_per_block, spatial_size, hist_bins, hog_channel=hog_channel)
        if bbox_list_curr:
            bound_box_list_all.append(bbox_list_curr)

        img_w_car_bbox = draw_boxes(img, bbox_list_curr)
        debug_plot.add_images(title='large cars on horizon - scale 2.0 : frame: {0}'.format(self.num_frame),
                              img_1=img_w_car_bbox, subtitle_1='img_w_car_bbox',
                              img_2=self.last_img_with_all_searched_bound_box, subtitle_2='img_w_all_bbox')


        bound_box_list_all = [item for sublist in bound_box_list_all for item in sublist]

        # -------------------------------------------------
        # Apply heat map
        # -------------------------------------------------
        # Add heat to each box in box list
        self.heat_map.add_heat(bound_box_list_all)
        # Apply threshold to help remove false positives
        heat = self.heat_map.apply_threshold(4)
        # Visualize the heatmap when displaying
        heat_map_img = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heat_map_img)

        self.last_heat_img = heat_map_img

        # self.save_images(img, heat_map_img, self.num_frame)
        self.num_frame += 1

        out_img = draw_labeled_bboxes(np.copy(img), labels)

        # -------------------------------------------------
        # DEBUGGING
        # -------------------------------------------------
        debug_plot.add_images(title='Summary',
                              img_1=cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), subtitle_1='Original img',
                              img_2=self.heat_map.get_heat_map_as_RGB(), subtitle_2='Heat map',
                              convert=False)
        # debug_plot.show_plot()

        out_img_w_boxes = draw_boxes(out_img, bound_box_list_all)
        img_w_overlay = add_thumbnail(out_img, out_img_w_boxes, x_offset=420)
        img_w_overlay = add_thumbnail(img_w_overlay, self.heat_map.get_heat_map_as_RGB())

        # return out_img
        return img_w_overlay

    def save_images(self, img, heatmap, frame):
        # plt.figure()
        # Plot the result
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.subplots_adjust(hspace=0.1, wspace=0.05)

        axes[0].imshow(img)
        title = "frame {0}".format(frame)
        axes[0].set_title(title, fontsize=11)
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].imshow(heatmap)
        title = "heatmap {0}".format(frame)
        axes[1].set_title(title, fontsize=11)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        plt.savefig('heat_maps/heatmap_{0}.jpg'.format(frame))

def train_and_return_svc(spatial, histbin, color_space, hog_channel, orient, pix_per_cell, cell_per_block):
    """

    :param spatial: (32, 32)
    :param histbin: 32
    :return:
    """
    # -------------------------------------------
    # Read in car and non-car images
    # -------------------------------------------
    cars = []
    notcars = []
    # cars
    images = glob.glob('vehicles/**/*.png', recursive=True)  # cars
    for image in images:
        cars.append(image)

    # non-cars
    images = glob.glob('non-vehicles/**/*.png', recursive=True)  # noncars
    for image in images:
        notcars.append(image)

    # TODO play with these values to see how your classifier
    # performs under different binning scenarios
    # spatial = 32
    # histbin = 32

    car_features = extract_features(cars, color_space=color_space, spatial_size=spatial, hist_bins=histbin,
                                    hog_channel=hog_channel, orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block)
    notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial, hist_bins=histbin,
                                    hog_channel=hog_channel, orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using spatial binning of:', spatial, 'and', histbin, 'histogram bins')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    return svc, X_scaler



def draw_boxes(img, bbox_list):
    draw_img = np.copy(img)
    for box in bbox_list:
        cv2.rectangle(draw_img, box[0], box[1], (255, 0, 0), 6)
    return draw_img

def add_thumbnail(img, thumb_img, scale=0.3, x_offset=2, y_offset=2):
    draw_img = np.copy(img)

    if thumb_img.ndim == 2:
        thumb_img = cv2.cvtColor((thumb_img / thumb_img.max()).astype('float32'), cv2.COLOR_GRAY2BGR)

    resized_thumb_img = cv2.resize(thumb_img, (0, 0), fx=scale, fy=scale)

    draw_img[y_offset:y_offset + resized_thumb_img.shape[0], x_offset:x_offset + resized_thumb_img.shape[1]] = resized_thumb_img
    return draw_img

def run_for_images():
    retrain = False
    if retrain:
        svc, X_scaler = train_and_return_svc(spatial=spatial_size, histbin=hist_bins, color_space=COLOR_SPACE,
                                             hog_channel=hog_channel, orient=orient, pix_per_cell=pix_per_cell,
                                             cell_per_block=cell_per_block)
        svc_pickle = {}
        svc_pickle['svc'] = svc
        svc_pickle['scaler'] = X_scaler
        pickle.dump(svc_pickle, open("svc_pickle.p", "wb"))
    else:
        dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
        svc = dist_pickle["svc"]
        X_scaler = dist_pickle["scaler"]

    images = glob.glob('test_images/*.jpg', recursive=True)  # cars
    for img_path in images:
        img = cv2.imread(img_path)

        veh_det = MasterVehicleDetection(img=img, svc=svc, X_scaler=X_scaler)

        out_img = veh_det.process_image(img)

        # Save image
        cv2.imwrite(os.path.join(r'output_images/', os.path.split(img_path)[1]), out_img)  # BGR

        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
        # plt.title('Car Positions')
        # plt.subplot(122)
        # plt.imshow(veh_det.last_heat_img, cmap='hot')
        # plt.title('Heat Map')
        # fig.tight_layout()
        # plt.show()

def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def run_for_video():
    clear_folder(r'heat_maps')

    retrain = False
    if retrain:
        svc, X_scaler = train_and_return_svc(spatial=spatial_size, histbin=hist_bins, color_space=COLOR_SPACE,
                                             hog_channel=hog_channel, orient=orient, pix_per_cell=pix_per_cell,
                                             cell_per_block=cell_per_block)
        svc_pickle = {}
        svc_pickle['svc'] = svc
        svc_pickle['scaler'] = X_scaler
        pickle.dump(svc_pickle, open("svc_pickle.p", "wb"))
    else:
        dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
        svc = dist_pickle["svc"]
        X_scaler = dist_pickle["scaler"]

    video_filename = 'test_video'
    # video_filename = 'project_video'
    video_output_filename = video_filename + '_output.mp4'

    # clip1 = VideoFileClip(video_filename + '.mp4').subclip(40, 45) # shadow
    clip1 = VideoFileClip(video_filename + '.mp4')

    img = cv2.imread(r'test_images/test1.jpg')
    veh_det = MasterVehicleDetection(img=img, svc=svc, X_scaler=X_scaler)

    white_clip = clip1.fl_image(veh_det.process_image)
    white_clip.write_videofile(video_output_filename, audio=False)

if __name__ == '__main__':
    # run_for_images()
    run_for_video()