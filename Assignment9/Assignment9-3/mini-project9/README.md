# Mini-Project 9: Traffic Sign Detection

## Team Members

- **Derek Zhang**
- **Yichi Zhang**
- **Yiling Hu**
- **Ziyang Liang**

---

## Project Overview

The **Stop Sign and Traffic Sign Detection** project is a real-time detection system that identifies stop sign and traffic signs from video feeds using **YOLOv5**. The system processes each video frame and detects traffic signs, specifically "Stop Signs" and "Traffic Signals." This project demonstrates the integration of a trained YOLOv5 model with **OpenCV** to detect and annotate objects in real time, with the option to save the output as a video file.

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

- Initial Dataset
  - We collected images for the following traffic sign categories from roboflow and manually changed the label:
    - Stop Sign (source: https://universe.roboflow.com/germantrafficsigns-zl3mn/stop-sign-zn1kw)
    - Traffic Signal (source: https://universe.roboflow.com/dat-uetjh/traffic-lights-lk2pn)
  - The dataset includes **478 training images** and **80 validation images**, with an equal number of images for each class.
- Finetune Dataset (manually collect)
  - From the first dataset, the model can't detect night traffic night, so we collect more data for this
    - source: [Google Drive Link](https://drive.google.com/drive/folders/1dwD1eU915KV71vdCAyP4SNgcF4uqavj9?usp=sharing)
  - The dataset includes **113 training images** and **10 validation images** and **6 test images**, with an equal number of images for each class.

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

---

### 1. Train the YOLOv5 Model

Use the `train.py` script provided in the YOLOv5 repository to train the model on our dataset: （please modify the default path to your downloaded data path)

```bash
python3 train.py --img 640 --batch 16 --epochs 50 --data '/content/placeholder/data.yaml' --weights yolov5s.pt 
```
This script:

- Loads the dataset and trains the model using YOLOv5.
- Saves the best-performing model for use in real-time detection.

## Best Model (download link)
[Google Drive Link](https://drive.google.com/file/d/1jZelWu5Ti8nAgi_RGXHweRAlaOucB5K8/view?usp=drive_link)
Model Evaluation:

- Detect Accuracy: 
  - Daytime: 90% 
  - NightTime: 60% (lack of green light data)
- Speed: 
  - For normal case is fast 
  - For edge case like dark and angle is bad is slower than nomal
- Challenge:
  - Night time data source including traffic light and stop sign is limited, causing the accuracy is low at night.
  - Computer gpu is not enough for training, when use the colab the training speed is less than before.

### 2. Run the Traffic Sign Detector

Launch the detection system using the `WebCamSave.py` script:

Since we are using google colab, it cannot compile cv2.imshow, so I remove the related lines of code.

```bash
python3 WebCamSave.py -f test.mp4 -o output_video.avi
```

---

## Demonstration Videos

Sample videos showing the detector in action are uploaded onto the repository or with Google Drive:

- **Recorded test video (**[Google Drive Link](https://drive.google.com/drive/folders/1eZgsuifq_x8-8hUAbe8NKM1LbDlg1hna?usp=drive_link)**):**
  - `mv1.mp4` → `mv1(1).mp4`
  - `night_video.mp4` → `night_video(1).mp4`
