## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_notcar.png
[image2]: ./examples/hogs.png

[image4]: ./examples/sliding_windows.jpg
[image41]: ./output_images/test1_multiple.jpg
[image42]: ./output_images/test2_multiple.jpg
[image43]: ./output_images/test3_multiple.jpg
[image44]: ./output_images/test4_multiple.jpg
[image45]: ./output_images/test5_multiple.jpg
[image46]: ./output_images/test6_multiple.jpg


[image5]: ./output_images/test1_full.jpg
[image6]: ./output_images/test2_full.jpg
[image7]: ./output_images/test3_full.jpg
[image8]: ./output_images/test4_full.jpg
[image9]: ./output_images/test5_full.jpg
[image10]: ./output_images/test6_full.jpg

[image11]: ./examples/bboxes_and_heat.png
[video1]: ./output_images/output_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! The entire project is completed in P5.ipynb

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the section under labeled: Feature extraction

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is a car and not car example example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. Based on theses gradients, it seems reasonable that a good feature set should incorporate these results.

![alt text][image2]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, the various color spaces and including spatial and/pr histogram binning. I tried to train my linear SVM with various combinations and kept an eye toward which parameters resulted in the best test accuracy (i.e. searched the paramater space).

I ultimately settled on only using HOG on the YUV colorspace (all the channels) with the parameters below achieveing a test accuracy of 98.17%

```
color_space = 'YUV' 

# Hog Parameters
orient = 11  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL"

```

This was quite annoying since each feature extraction (for all images) took ~1 minute.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

"Train Linear SVM Classifier" section holds the code I used for training my linear SVM with the final paramters above. I only used the HOG features and therefore did not bother normalizing.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

"Sliding window search" section holds the code for scanning the image. 

Given that objects farther away appear smaller, I decided to have smaller windows for horizontal bands nearer the top of the image (code found in "Various window sizes" section). I achieved the various bands by using different `ystart` and `ystop` parameters for image to define the horizontal band. To use get different size window i use the `scale` paramater to my `find_cars()` (i.e. larger scale means larger window, scale = 1.0 => window of 64x64)

I also ignored all the sky and car hood (global `ystart=400, ystop=600`

All Search Windows found here (generated:
![alt text][image4]

And the results:
![alt text][image41]
![alt text][image42]
![alt text][image43]
![alt text][image44]
![alt text][image45]
![alt text][image46]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on using YUV 3-channel HOG features, which provided a nice result. 

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image8]
![alt text][image9]
![alt text][image10]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/output_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Code found in the "Heatmap" section.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image11]

I also implemented a way to take into account several frames of the video at once before drawing the bounding boxes. Code found in `VehicleFrame() class`. The goal was to ignore any one spurious frame by increasing the threshold as history of frames is built up.

```
class VehicleFrame:
    def __init__(self, queue_size=10):
        self.queue_size = queue_size
        self._box_lists = []
    
    def add_box_list(self, box_list):
        self._box_lists = [box_list] + self._box_lists
        self._box_lists = self._box_lists[:self.queue_size]
    
    @property
    def buffer_length(self):
        return len(self._box_lists)
    
    def get_new_threshold(self, threshold):
        return threshold + self.buffer_length // 2
    
    @property
    def box_list(self):
        # flatten list
        return [item for sublist in self._box_lists for item in sublist]
```


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My implementation isn't that robust, inspite of removing as many false positives as possible I still end up several and the bounding boxes around the cars are not super tight. I would spend a lot more time refining the search windows since the training itself was reasonably accurate. I would potentially also modify the search windows based on previous results (i.e if i noticed a "large" car in the lower half of the image, then look for a slightly smaller car a little higher in the image a few frames later) 
