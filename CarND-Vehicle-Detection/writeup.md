##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_example.png
[image5]: ./output_images/heatmap.png
[pipeline1]: ./output_images/pipeline1.png
[pipeline2]: ./output_images/pipeline2.png
[pipeline3]: ./output_images/pipeline3.png
[pipeline4]: ./output_images/pipeline4.png
[pipeline5]: ./output_images/pipeline5.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


###Histogram of Oriented Gradients (HOG)

####1. How I extract HOG features from the training images.

The code for this step is contained in the 8th code cell of the IPython notebook .  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. How I settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally choice ,`pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.
Comparing the Hog output of the vehicle image and notvehicle image, I chose the parameter that makes a clear difference.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features ,spatial features and color histgram features.
The code for this step is contained in the 9th code cell of the IPython notebook .  
Comparing the accuracy with spatial faetures and color histgram features, the accuracy was the best when using both.　　

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Since the upper half of the image is sky, I decided not to search.
Also, as you got closer to the bottom of the image, the car got bigger, so the size of the window was made larger.
I search with windows of multiple scales.  
The code for this step is contained in the 9th code cell of the IPython notebook .  


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][pipeline1]
![alt text][pipeline2]
![alt text][pipeline3]
![alt text][pipeline4]
![alt text][pipeline5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  



![alt text][image5]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In several frames, an incorrect result was output.
Since it is considered that the car does not move greatly during one frame, the heat map of the previous frame is added with coefficient to the current search result.
As a result, it is possible to ignore the instantaneous false.

