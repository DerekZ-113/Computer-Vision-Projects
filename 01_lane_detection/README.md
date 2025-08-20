# 01 Lane Detection for Autonomous Vehicles

## Project Description

This project implements a **Lane Detection System** specifically designed for **autonomous vehicle applications**, using advanced computer vision techniques to enable safe navigation and path planning. The system serves as a critical component of the autonomous driving perception pipeline, providing real-time lane boundary detection that allows self-driving vehicles to:

🚗 **Stay within lane boundaries** during normal driving conditions  
🛣️ **Follow road geometry** and adapt to curved roads  
⚠️ **Detect lane departures** for safety alerts and corrective actions  
🎯 **Support path planning algorithms** with accurate road structure information  

The lane detection system processes video frames in real-time to identify lane boundaries using **color filtering**, **edge detection**, and **Hough Transform**, ensuring robust performance even when lane markings are partially obscured, faded, or consist of dashed lines. This technology is fundamental to **lane-keeping assistance systems** and **autonomous navigation**.

## Autonomous Vehicle Applications

This lane detection system addresses several critical autonomous vehicle challenges:

### 🛡️ **Safety & Navigation**
- **Lane Keeping Assistance**: Prevents unintentional lane departures
- **Steering Control**: Provides lane center information for automated steering
- **Road Boundary Detection**: Identifies safe driving areas vs. shoulders/medians

### 🎯 **Path Planning Support**
- **Lane Center Calculation**: Determines optimal vehicle positioning
- **Curvature Detection**: Identifies road curves for speed adjustment
- **Intersection Handling**: Detects lane convergence points for navigation decisions

### 🔄 **Real-time Performance**
- **Low Latency Processing**: Essential for real-time driving decisions
- **Robust Detection**: Handles various lighting conditions and road surfaces
- **Continuous Monitoring**: Provides consistent lane information for autonomous systems

## Features

- **Real-time autonomous vehicle perception**: Designed for integration with autonomous driving systems requiring immediate lane boundary information for navigation decisions.
- **Multi-input support**: 
  - **Video file processing** for testing and validation of autonomous vehicle algorithms
  - **Live webcam feed** for real-time autonomous vehicle simulation and development
- **Robust lane detection optimized for autonomous driving**:
  - HSV-based white color filtering to detect lane markings in various lighting conditions (HSV range: [0,0,220] to [180,25,255])
  - Handles challenging scenarios common in autonomous driving: faded markings, construction zones, and weather conditions
- **Automotive-grade edge detection**: Uses Canny edge detection (thresholds: 50-150) optimized for road surface analysis and lane marking identification
- **Precision line detection**: Hough Line Transform (rho=1, theta=π/180, threshold=50) specifically tuned for detecting left and right lane boundaries with automotive accuracy requirements
- **RANSAC outlier removal**: Identifies the most reliable lane lines by removing sensor noise and false detections, critical for autonomous vehicle safety
- **Temporal smoothing for autonomous vehicle stability**: 
  - Uses buffering system (deque with maxlen=30) to store lane lines from previous frames
  - Averages lines over multiple frames to reduce steering instability
- **Autonomous vehicle visualization**:
  - Real-time lane overlay with red lines (10px thickness) for driver assistance systems
  - Recorded output for autonomous vehicle testing and validation

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

## Usage Guide for Autonomous Vehicle Development

- **Autonomous Vehicle Integration**:
  - The system outputs lane boundary coordinates that can be directly integrated with autonomous vehicle path planning algorithms
  - Lane center calculations support steering control systems
  - Real-time performance suitable for automotive safety requirements (low latency, high reliability)
- **Testing & Validation**:
  - Use recorded video files to test lane detection performance across various driving scenarios
  - Live webcam testing for real-time autonomous vehicle algorithm development
- **Controls**:
  - The program automatically starts processing frames for continuous autonomous vehicle operation
  - Press `s` to Pause/Resume (useful for debugging autonomous vehicle algorithms)
  - Press `q` to Quit the application
- **Automotive Requirements**: Ensure your system meets automotive-grade processing requirements with sufficient computational power for real-time lane detection in autonomous vehicle applications

## Code Overview - Autonomous Vehicle Architecture

- **`color_filter`**: Automotive-grade HSV color filtering to isolate white lane markings under various lighting conditions. Uses HSV range [0,0,220] to [180,25,255] for robust white detection in autonomous driving scenarios.

- **`region_of_interest`**: Defines the vehicle's field of view using a trapezoidal mask focusing on the road area. Creates polygon vertices at [(0,h), (w,h), (0.8w,0.4h), (0.2w,0.4h)] to concentrate processing on relevant autonomous navigation areas.

- **`line_filter_pipeline`**: Core autonomous vehicle perception pipeline that processes detected lines by:
   - Filtering lines based on automotive slope criteria (removes irrelevant road features)
   - Applying RANSAC for automotive-grade outlier removal and lane line selection
   - Implementing temporal smoothing using 30-frame buffering for stable autonomous vehicle control

- **`line_slope_filter`**: Automotive-specific slope filtering that removes lines with slopes between -0.3 and 0.3 (near-horizontal markings like crosswalks, stop lines) and focuses on lane boundaries relevant to autonomous navigation.

- **`RANSAC`**: Implements robust lane detection using RANSAC algorithm with distance threshold of 10 pixels and 100 max iterations to handle sensor noise and ensure reliable lane identification for autonomous vehicle safety systems.

- **`find_best_match_line`**: Autonomous vehicle lane classification system that:
  - Separates detected lines into left (negative slope) and right (positive slope) lane candidates
  - Uses slope difference threshold of 0.15 and minimum distance threshold of 30 pixels
  - Essential for maintaining vehicle position within the lane

- **`add_to_buffer` & `average_lines`**: Temporal filtering system using deque buffers (maxlen=30) that provides smooth, stable lane detection for autonomous vehicle control systems, reducing frame-to-frame fluctuations that could cause steering instability.

- **`draw_lines`**: Visualization system for autonomous vehicle development and testing, overlaying detected lane boundaries in red (10px thickness) for validation and debugging.

- **`make_line_coordinates`**: Generates precise lane boundary coordinates for autonomous vehicle path planning:
  - Extends lane lines from bottom of image (y=h) to intersection vicinity (y=iy+0.3h, capped at 0.8h)
  - Uses slope-intercept form for accurate coordinate calculation

- **`line_intersection`**: Calculates lane convergence points for autonomous vehicle navigation at intersections and lane merges using linear algebra intersection formulas.

- **`line_point_distance` & `line_distance`**: Precision measurement functions for autonomous vehicle positioning within lanes and maintaining safe distances from lane boundaries using perpendicular distance calculations.

- **`main`**: Autonomous vehicle perception main loop that:
  - Processes video at 20 FPS output using XVID codec
  - Applies Gaussian blur (5x5 kernel) and Canny edge detection (50-150 thresholds)
  - Uses Hough Line Transform with automotive-optimized parameters (minLineLength=50, maxLineGap=200)
  - Provides pause/resume functionality for debugging autonomous vehicle algorithms
  - Outputs lane detection results for integration with autonomous vehicle control systems

## Autonomous Vehicle Testing & Validation

- **Sample videos in Google Drive:**
  [Google Drive Link](https://drive.google.com/drive/folders/1MfBh7xroQ3lGHEUabCWor48rcw4rfypk?usp=drive_link)

  - `lane_test1.mp4` → `out_lane_test1.avi` & `Sample for test1.mov`
  - `lane_test2.mp4` → `out_lane_test2.avi` & `Sample for test2.mov`
  - `lane_test3.mp4` → `out_lane_test3.avi` & `Sample for test3.mov`

- **Real-world autonomous vehicle testing:**
  
  **Note:** Testing was conducted using WebCam and DashCam footage to simulate real autonomous vehicle scenarios.

  - `real-road-1.mp4` → `out_real-road-1.avi`
  - `real-road-2.mp4` → `out_real-road-2.avi`
  - `real-road-3.mp4` → `out_real-road-3.avi`

- **Autonomous Vehicle Scenarios Tested:**
  - **Urban driving**: Complex road markings and intersection navigation
  - **Highway driving**: High-speed lane detection and lane change scenarios  
  - **Varied lighting conditions**: Dawn, dusk, and nighttime autonomous driving
  - **Weather conditions**: Testing robustness for autonomous vehicle operation in different weather
