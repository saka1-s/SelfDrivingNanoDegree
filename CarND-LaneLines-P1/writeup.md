**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./improve.png "Improve"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, 
then I applyed the gaussian filter to image.   
Next, by extracting the canny edge, 
I extracted a part with large luminance change from the image.  
Next, I specified a range to detect lane lines. 
Finally, Hough transform was performed to calculate the slope and bias of the lane lines.

In order to draw a single line on the left and right lanes, 
I modified the draw_lines() function by using polynomial curve fitting and
moving average.  
Polynomial curve fitting is used to find one average line segment from a plurality of line segments and
moving average is used to obtain robustness against noise.



###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the color of the road and the line are similar.

Another shortcoming could be happened when the car would change lane.


###3. Suggest possible improvements to your pipeline

A possible improvement would be dynamically change parameters according to circumstances.  
Another potential improvement could be to predict the current situation from past information.
![alt text][image1]
