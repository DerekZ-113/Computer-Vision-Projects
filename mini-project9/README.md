# Mini-Project 9: Traffic Sign Detection

## Team Members

- **Derek Zhang**
- **Yichi Zhang**
- **Yiling Hu**
- **Ziyang Liang**

---

## Project Overview

The **Traffic Sign Detection** project is a real-time detection system that identifies traffic signs from video feeds using **YOLOv5**. The system processes each video frame and detects traffic signs, specifically "Stop Signs" and "Traffic Signals." This project demonstrates the integration of a trained YOLOv5 model with **OpenCV** to detect and annotate objects in real time, with the option to save the output as a video file.

---

## Features

- **Real-time Traffic Sign Detection:**
  - Detects "Stop Sign" and "Traffic Signal" in each frame of the video feed with high accuracy.
  - Displays bounding boxes and confidence scores directly on the video feed.
- **Video and Webcam Integration:**
  - Uses OpenCV to capture frames from a video file or connected webcam.
- **Confidence Overlay:**
  - Shows the detected object labels and confidence scores on each frame.
- **Modular Model Architecture:**
  - Easy to adjust for different traffic signs or add additional classes.

---

## Setup Instructions (Google Colab)

### 1. Environment Setup

Ensure that the runtime is switched to T4 GPU (We are using free version).

And setup yolov5 environment by running the following code.

```bash
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
```

### 2. Dataset Preparation

- We collected images for the following traffic sign categories from roboflow and manually changed the label:
  - Stop Sign (source: https://universe.roboflow.com/germantrafficsigns-zl3mn/stop-sign-zn1kw)
  - Traffic Signal (source: https://universe.roboflow.com/dat-uetjh/traffic-lights-lk2pn)
- The dataset includes **478 training images** and **80 validation images**, with an equal number of images for each class.

```bash
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

- **Data Annotation:** Each image was annotated using bounding boxes for YOLOv5 compatibility.
- **Data Source:** The dataset can be downloaded from this GitHub repository, or from the following link. Ensure that the dataset is unzipped in the same project folder for smooth execution:
  [Google Drive Link](https://drive.google.com/file/d/1BBV0IBQMYGVgzXych-46r6Pmu3uuWLtb/view?usp=sharing)

---

## Running the Application

### 1. Train the YOLOv5 Model

Use the `train.py` script provided in the YOLOv5 repository to train the model on our dataset:

```bash
python3 train.py --img 640 --batch 16 --epochs 50 --data '/content/dataset/data.yaml' --weights yolov5s.pt --name traffic_detection
```

This script:

- Loads the dataset and trains the model using YOLOv5.
- Saves the best-performing model for use in real-time detection.

### 2. Run the Traffic Sign Detector

Launch the detection system using the `WebCamSave.py` script:

Since we are using google colab, it cannot compile cv2.imshow, so I remove the related lines of code.

```bash
python3 WebCamSave.py -f test.mp4 -o output_video.avi
```

---

## Demonstration Videos

Sample videos showing the detector in action are uploaded onto the repository or with Google Drive:

- **Recorded test video (**[Google Drive Link](https://drive.google.com/file/d/16h6gAzWkbrao9sI6SV5htQ4BfZkJP_f0/view?usp=sharing)**):**
  - `mv1.mp4` → `output_video1.avi`
  - `mv2.mp4` → `output_video2.avi`
  - `mv3.mp4` → `output_video3.avi`
