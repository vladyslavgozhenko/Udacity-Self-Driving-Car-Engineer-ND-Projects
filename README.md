# **Finding Lane Lines on the Road** 


<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

## Overview

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project I will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

---
## Pipeline 

1. To extract lane lines from still images will be used grayscalling, Gaussian smoothing, color selection, region of interest selection, Canny Edge Detection and Hough Transform, approximation/extrapolation of straight lines. On the following steps are show results of applying different functions to an image.  

<img src="https://github.com/wiwawo/CarND-Term1/blob/CarND-LaneLines-P1-master/pipeline.png" width="480" alt="Combined Image" />

Red lines show found the left and right lanes, green lines show the region of interest. In the algorithm was analysed only the region of interest for the possible lanes on it. 

2. To extract lanes from video files video files, I will apply steps from the first paragraph to each video frame. To avoid shaking of annotated lanes on videos I will smooth annotated lines from several frames (some kind of moving average of lane lines between adjacent frames). Number of smoothing parameters was chosen to avoid shaky lines.
3. In the challenge, I extracted "average" color from a road (in the test video it was grayish color). After that I added tolerances to select all the road colors, except for the white lanes or yellow side lanes. This approach helped me overcome effect on lane colors/brightness from different illumination of the highway, because the illumination changed "overage" color of the road as well. 

---

## Shortcomings

The algorithm won't work/won't be efficient if the following cases:
* if a road very curvy, since Hough Transform won't find any lines there;
* the algorithm doesn't take into account perspective (so all the line always distorted and not really parallel) and camera position, it can explain shaky on videos.
* a problem could be weather conditions: too much water or snow on the road. Snow can be interpreted as line and water can hide the lanes;
* temporal and regular lanes can be mixed up (i.e. during high way repairs).

---

## Possible improvements
* use other points of interest to identify exact position of the lanes (trees, other cars, high way concrete boarder etc.);
* use additionally some other non optical methods of identifying lanes (laser or some kind of radar);
* use other geometrical approach: take into account geometrical distortions due the perspective. Take into account exact camera position or use even 2 cameras.  

---
