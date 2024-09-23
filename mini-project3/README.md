# Webcam Transformation Application

**Group Member** Yichi Zhang, Yiling Hu, Ziyang Liang, Derek Zhang

This application is a simple webcam-based tool that allows users to apply real-time transformations such as translation, rotation, scaling, and perspective to the video feed from their webcam. The app is built using OpenCV and Python.

## Features

- **Real-time Video Feed**: The application captures a live video feed from the user's webcam and allows transformations on the fly.
- **Transformation Modes**:
  - **Translation**: Shift the video frame horizontally or vertically.
  - **Rotation**: Rotate the video frame by a specified degree.
  - **Scaling**: Resize the video frame by a scaling factor.
  - **Perspective**: Apply a perspective transformation to simulate a different viewpoint.
- **Toggle Between Modes**: Switch between different transformation modes using keyboard input(pressing the same key again will return to the original mode):
  - `t`: Translation
  - `r`: Rotation
  - `s`: Scaling
  - `p`: Perspective
- **Original and Transformed Views**: The application shows both the original frame and the transformed frame side by side.
- **FPS Display**: The frames per second (FPS) is displayed on both the original and transformed frames.

## How to Use

1. You can switch between transformation modes by pressing the following keys:
   - `t`: Apply/Cancel translation to the frame.
   - `r`: Apply/Cancel rotation to the frame.
   - `s`: Apply/Cancel scaling to the frame.
   - `p`: Apply/Cancel perspective transformation.
   - `q`: Quit the application.

2. Both the original and transformed video frames will be displayed side by side.

