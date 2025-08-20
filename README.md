# Computer Vision Projects - CS5330 2024 Fall

## Table of Contents

- [Computer Vision Projects - CS5330 2024 Fall](#computer-vision-projects---cs5330-2024-fall)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Autonomous Vehicle Vision Pipeline](#autonomous-vehicle-vision-pipeline)
  - [Technologies Used](#technologies-used)
  - [Detailed Project Descriptions](#detailed-project-descriptions)
    - [01 Lane Detection](#01-lane-detection)
    - [02 Traffic Sign Detection](#02-traffic-sign-detection)
    - [03 Road Segmentation](#03-road-segmentation)
    - [04 Object Tracking](#04-object-tracking)
  - [General Setup Instructions](#general-setup-instructions)

## Project Overview

This repository showcases a comprehensive computer vision system designed for **autonomous vehicle applications**, developed as part of our coursework for CS5330 Computer Vision, Fall Semester 2024. Our four interconnected projects collectively address the core visual perception challenges that self-driving cars must solve to navigate safely and effectively in real-world environments.

Each project tackles a fundamental aspect of autonomous driving:

🛣️ **Lane Detection** - Enables the vehicle to stay within lane boundaries and follow road geometry.  
🚦 **Traffic Sign Recognition** - Allows the vehicle to understand and obey traffic regulations.  
🛤️ **Road Segmentation** - Helps the vehicle distinguish drivable areas from obstacles and hazards.  
👁️ **Object Tracking** - Enables the vehicle to monitor and predict the movement of dynamic objects like pedestrians and other vehicles.

Together, these projects form a cohesive vision system that demonstrates how multiple computer vision techniques work in harmony to create the visual intelligence required for autonomous navigation. Each component is essential for safe autonomous driving, as vehicles must simultaneously understand road structure, follow traffic rules, identify safe driving areas, and track moving objects in their environment.

## Autonomous Vehicle Vision Pipeline

Our projects simulate the key components of an autonomous vehicle's perception system:

```
🎥 Camera Input
    ↓
📍 Lane Detection (Navigation & Path Planning)
    → Identifies lane boundaries for steering control
    → Provides road curvature information
    → Enables lane-keeping assistance
    ↓
🚦 Traffic Sign Detection (Regulatory Compliance)
    → Recognizes stop signs and traffic signals
    → Enables rule-based decision making
    → Supports intersection navigation
    ↓
🛤️ Road Segmentation (Environment Understanding)
    → Distinguishes drivable vs. non-drivable areas
    → Identifies road surfaces and boundaries
    → Supports path planning algorithms
    ↓
👥 Object Tracking (Dynamic Scene Analysis)
    → Monitors pedestrians, vehicles, and obstacles
    → Predicts object trajectories
    → Enables collision avoidance systems
    ↓
🚗 Autonomous Driving Decision System
```

This integrated approach mirrors how real autonomous vehicles process visual information, where each component provides critical data that informs the vehicle's understanding of its environment and influences driving decisions.

## Technologies Used

- **Deep Learning Frameworks**: TensorFlow, PyTorch, Ultralytics YOLOv5/YOLOv8
- **Computer Vision Libraries**: OpenCV, NumPy, SciPy
- **Machine Learning**: Custom U-Net architecture, RANSAC algorithm, Lucas-Kanade Optical Flow
- **Image Processing**: Canny edge detection, Hough Transform, morphological operations
- **Data Visualization**: Matplotlib, real-time video processing

## Detailed Project Descriptions

### 01 Lane Detection

**Advanced lane detection system using computer vision techniques**

**Features:**
- Real-time lane detection from video files or live webcam feed
- Robust edge detection using Canny edge detection
- Color filtering to enhance white lane line detection
- RANSAC algorithm for outlier removal and best lane line selection
- Hough Transform for accurate line detection
- Morphological operations to connect dashed lines
- Frame buffering and averaging for smooth, stable detection

**Key Technologies:**
- OpenCV for image processing
- RANSAC for robust line fitting
- Hough Line Transform
- Color space filtering and edge detection

**Setup:**
```bash
pip install opencv-python numpy
python WebCamSave.py -f video_file_name -o out_video.avi
```

**Controls:**
- Press `s` to Pause/Resume
- Press `q` to Quit

**Demo Videos:** [Google Drive Link](https://drive.google.com/drive/folders/1MfBh7xroQ3lGHEUabCWor48rcw4rfypk?usp=drive_link)

---

### 02 Traffic Sign Detection

**Real-time traffic sign detection using YOLOv5**

**Features:**
- Detects "Stop Sign" and "Traffic Signal" with high accuracy
- Real-time processing with confidence score overlay
- Supports both video files and webcam input
- Modular architecture for easy class expansion
- Optimized for both daytime and nighttime conditions

**Model Performance:**
- **Daytime Accuracy**: 90%
- **Nighttime Accuracy**: 60% (limited by training data)
- **Dataset**: 478 training images, 80 validation images

**Setup (Google Colab):**
```bash
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt

# Training
python3 train.py --img 640 --batch 16 --epochs 50 --data '/content/placeholder/data.yaml' --weights yolov5s.pt

# Detection
python3 WebCamSave.py -f test.mp4 -o output_video.avi
```

**Model Downloads:**
- [Base Model](https://drive.google.com/file/d/1BBV0IBQMYGVgzXych-46r6Pmu3uuWLtb/view?usp=sharing)
- [Fine-tuned Model](https://drive.google.com/file/d/1ny4jpXZBfa-oN0bZNR4hRDi9sVtU-3Gu/view?usp=sharing)

**Demo Videos:** [Google Drive Link](https://drive.google.com/drive/folders/1eZgsuifq_x8-8hUAbe8NKM1LbDlg1hna?usp=drive_link)

---

### 03 Road Segmentation

**Semantic road segmentation using customized U-Net architecture**

**Features:**
- Custom U-Net model for pixel-level road segmentation
- Advanced data augmentation (rotation, zoom, brightness adjustment)
- Combination loss function (Binary Cross-Entropy + Dice Loss)
- IoU and accuracy metrics for comprehensive evaluation
- Optimized hyperparameters for urban road scenes

**Model Architecture:**
- **Input Shape**: 256x256x3
- **Batch Size**: 8
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Binary Cross-Entropy + Dice Loss
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

**Results (15 epochs):**
- **Training Accuracy**: 94.94%
- **Training IoU**: 86.98%
- **Validation Accuracy**: 93.03%
- **Validation IoU**: 81.49%

**Setup:**
```bash
pip install tensorflow opencv-python-headless numpy matplotlib scikit-learn
```

**Dataset:** [Cityscape Dataset](https://drive.google.com/drive/folders/1qRVz70ixvsj76Pp2Gwv982A6ZFzYPG-o) (1391 images with ground truth masks)

**Training Files:** Available in [Google Drive](https://drive.google.com/drive/folders/1qRVz70ixvsj76Pp2Gwv982A6ZFzYPG-o)

---

### 04 Object Tracking

**Real-time object detection and tracking system**

**Features:**
- YOLOv8 for comprehensive object detection
- Lucas-Kanade Optical Flow for object tracking
- Selective tracking of specified classes (e.g., 'person')
- Real-time FPS monitoring
- Interactive controls for tracking management
- Multi-object tracking with trajectory visualization

**Interactive Controls:**
- Press `t` to toggle tracking on/off
- Press `c` to clear tracking data and trajectories
- Press `q` to quit application

**Technical Details:**
- **Detection Model**: YOLOv8 (nano, small, medium, large, extra-large variants)
- **Tracking Algorithm**: Lucas-Kanade Optical Flow
- **Performance Optimization**: Detection every 3 frames, frame resizing
- **Visualization**: Green trajectory lines, yellow tracking boxes, blue detection boxes

**Setup:**
```bash
pip install ultralytics opencv-python scipy numpy
python3 webcam.py
```

**Performance Features:**
- FPS display for performance monitoring
- Optimized frame processing for real-time performance
- GPU acceleration support
- Configurable detection frequency and frame resolution

**Demo Videos:** [Google Drive Link](https://drive.google.com/drive/folders/10ehLSJHL3uk9QPhVTZnkJCsxqea7fR6Z?usp=share_link)

## General Setup Instructions

**System Requirements:**
- Python 3.6 or higher
- OpenCV 4.x
- NumPy, SciPy, Matplotlib
- TensorFlow (for U-Net projects)
- Ultralytics (for YOLO projects)

**Common Dependencies:**
```bash
pip install opencv-python numpy scipy matplotlib tensorflow ultralytics
```

**Hardware Recommendations:**
- GPU with CUDA support (recommended for deep learning projects)
- Minimum 8GB RAM
- Webcam or video files for testing