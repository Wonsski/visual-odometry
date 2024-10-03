# Visual Odometry

Visual Odometry (VO) is the process of estimating the motion of a camera by analyzing a sequence of images. This repository provides a basic implementation of monocular visual odometry using OpenCV's ORB feature detector and matcher. It is designed to work with datasets like KITTI, and it estimates the camera's movement based on visual information from consecutive frames.


https://github.com/user-attachments/assets/61547a31-226d-44d5-ba1a-4e75350cddc1




## Features
- ORB (Oriented FAST and Rotated BRIEF) keypoint detection and description.
- Keypoint matching with the BFMatcher (Brute Force Matcher) using the Hamming distance.
- Essential matrix computation and pose recovery to estimate the camera’s motion (rotation and translation).
- Visualization of keypoints and feature matches between consecutive frames.
- Simple trajectory visualization on a 2D plane.
- Motion threshold to filter out small movements and reduce noise.

## Installation

### Prerequisites
- Python 3.x
- OpenCV 4.x
- Numpy
