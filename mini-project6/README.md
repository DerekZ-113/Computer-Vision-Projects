# Mini-Project6: Lane Detector

### Project Group 4

- Derek Zhang
- Yichi Zhang
- Yiling Hu
- Ziyang Liang

## Project Description

This project implements a **Lane Detector** using advanced image processing techniques to detect lane lines on the road. The system processes video frames to identify lane boundaries, enabling the detection of the left and right lane lines. The project uses **color filtering**, **edge detection**, and **Hough Transform** to accurately identify lane lines, even when they are partially obscured or faded.

## Features

- **Video file or live webcam feed**: Supports both pre-recorded video files and live feed from a connected camera for lane detection.
- **Real-time lane detection**:
  - Automatically processes each frame to detect lane lines in real time.
  - Utilizes color filtering to enhance the detection of lane lines on different road conditions.
- **Robust edge detection**: Uses Canny edge detection to identify significant changes in pixel intensity, helping to isolate lane lines.
- **Lane enhancement**: Applies morphological operations to strengthen the detection of dashed white lines by connecting them.
- **Hough Transform for line detection**: Uses the Hough Line Transform to detect and draw the left and right lane lines separately.
- **Output visualization**:
  - Displays the lane detection results on the video in real time.
  - Saves the processed video with detected lane lines for later review.

## Set Up Instructions

1. **Install Dependencies:**
   - You need Python 3.x and OpenCV.
   - Install OpenCV via pip:
     ```bash
     pip install opencv-python numpy
     ```
2. **Running the Application:**
   - Run the script with a video file or live webcam:
     ```bash
     python WebCamSave.py -f video_file_name -o out_video.avi
     ```

## Usage Guide

- **Controls**:
  - The program automatically starts processing frames as soon as it runs.
  - Press `q` to quit the application and stop the lane detection.
- **Ensure your computer has a connected webcam or provide a valid video file**. The application processes each video frame in real time, applying image enhancement techniques to improve the detection of lane lines, even in challenging lighting conditions or with dashed lines.

## Code Overview

- **`enhance_white_lines`**: Converts the image to grayscale and applies adaptive thresholding and morphological operations to enhance and connect broken dashed white lines, making them easier to detect.
- **`color_filter`**: Filters the image to isolate the white lane lines, improving the accuracy of lane detection by reducing noise and irrelevant elements.
- **`region_of_interest`**: Defines a polygonal region of interest to focus the detection on the area where lane lines are most likely to appear.
- **`average_slope_intercept`**: Calculates the average slope and intercept for the left and right lane lines, allowing for the detection of smoother and more consistent lane boundaries.
- **`make_line_coordinates`**: Generates line coordinates for the left and right lane lines based on their average slope and intercept.
- **`display_lines`**: Draws the detected lane lines onto the video frames, showing them in real time with clear visual indicators.
- **`main`**:
  - Initializes the video feed and processes each frame using the lane detection pipeline.
  - Enhances the image to detect dashed lines as solid, continuous lines for improved visualization.
  - Outputs the processed video with lane markings.

### Demonstrating Video

- Sample video in the project folder
  - lane_test1.mp4 -> out_lane_test1.avi
  - lane_test2.mp4 -> out_lane_test2.avi
  - lane_test3.mp4 -> out_lane_test3.avi
- Optional recorded video:
