
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[imageCal]: ./output_images/calibration3.jpg "Raw Image"
[image1]: ./output_images/Undistorted.jpg "Undistorted"
[image2]: ./output_images/img.png "Road Transformed"
[image3]: ./output_images/img_threshold.png "Binary Example"
[image4_origin]: ./output_images/orign_img.png "Warp Example"
[image4_wrap]: ./output_images/wraped_img.png "Warp Example"
[image6]: ./output_images/result.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "./project.ipynb"   

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.  
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][imageCal]
![alt text][image1]

### Pipeline (single images)

#### 1. How to correct image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
The code for this step is contained in the second code cell of the IPython notebook located in `./project.ipynb`   
I applied the calculated camera calibration data to image and got an undistorted image.

![alt text][image2]

#### 2. How I use color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cells 3 through 14 in `./project.ipynb`).  
Here's an example of my output for this step.  I tweaked thresholds by using GUI.
![alt text][image3]

#### 3. How I performe a perspective transform 

The code for my perspective transform includes a function called `wrap()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
# "offset_px" is the pixel of center of the lane. 
src = np.float32(
    [[img.shape[1]/2 - offset_px+70,img.shape[0]*0.65],
    [img.shape[1]/2 - offset_px+400,img.shape[0]*0.9],
    [img.shape[1]/2 - offset_px-400,img.shape[0]*0.9],
        [img.shape[1]/2 - offset_px-70,img.shape[0]*0.65]])
dst = np.float32(
    [[1030,100],
    [1030,720],
    [250,720],
        [250,100]])
```

This resulted in the following source and destination points when offset_px is 0:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 710, 460      | 1030, 100        | 
| 1040, 720      | 1030, 720      |
| 240, 720     | 250, 720      |
| 570, 460      | 250, 100        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4_origin]
![alt text][image4_wrap]

#### 4. How I identify lane-line pixels and fit their positions with a polynomial

I did this in cell 24 in my code in `project.ipynb`


#### 5. How I calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 68 through 72 in cell 26 in `project.ipynb`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cells 27 in my code in `project.ipynb` .  
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

If the car falls on one side of the lane, the perspective transform may not work correctly.  
Therefore, the positional relationship between the vehicle and the lane is estimated from the calculation result of the previous frame.
And I add an offset to the perspective transform.
