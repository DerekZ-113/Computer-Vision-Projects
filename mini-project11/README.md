# Mini-Project 11 Follow Everything

## Team Members

- **Derek Zhang**
- **Yichi Zhang**
- **Yiling Hu**
- **Ziyang Liang**

---

## **Project Overview**

This project demonstrates a Python-based real-time object detection and tracking application using:

- **YOLOv5**: For object detection.
- **Lucas-Kanade Optical Flow**: For tracking detected objects across video frames.

The goal is to detect objects using a pre-trained YOLOv5 model and track their movements using optical flow, enabling continuous monitoring of selected regions.

---

## **Features**

1. **Object Detection**:
   - Utilizes the pre-trained YOLOv5 model for detecting objects in a video stream or file.
   - Draws bounding boxes with class labels and confidence scores for detected objects.
2. **Manual ROI Selection**:
   - Users can manually select a region of interest (ROI) for tracking specific objects.
3. **Optical Flow Tracking**:
   - Tracks the manually selected ROI across frames using Lucas-Kanade optical flow.
4. **Video Input Options**:
   - Supports real-time webcam input or pre-recorded video files.
5. **Video Output**:
   - Saves the processed video with detection and tracking annotations.

---

## **Setup Instructions**

1.  **Install Required Libraries**:
    Install the dependencies listed below:
        ```bash
        pip install ultralytics opencv-python argparse numpy
        ```
2.  **Download Pre-trained YOLOv5 Model**:
    We are downloading the `yolov5s.pt` model due to device limitation. But you can switch to all these models if available.
        ```python
        # yolov5s.t: (fastest, least accurate)
        # yolov5m.pt: Medium
        # yolov5l.pt: Large
        # yolov5x.pt: Extra-large (most accurate, slowest)
        yolo_model = YOLO('yolov5s.pt')
        ```
3.  **Run the Script**:
    Execute the script with the following command:
        ```bash
        python3 webcam.py -f <video-file> -o <output-video>
        #python3 webcam.py -f test.mp4 -o output_video.avi
        ```

        - Replace `<video-file>` with the path to a video file (optional).
        - Replace `<output-video>` with the desired output video file name (optional).
        - If no `f` option is provided, the webcam will be used as the input.

---

## **Usage Instructions**

1. **Object Detection**:
   - The script continuously detects objects and displays bounding boxes with labels.
2. **Manual ROI Selection**:
   - Press `s` to manually select a region of interest (ROI) for tracking.
   - Use the mouse to draw a bounding box around the object.
3. **Optical Flow Tracking**:
   - Once an ROI is selected, the Lucas-Kanade optical flow algorithm will track the object.
   - Tracks are displayed as green lines.
4. **Exit**:
   - Press `q` to quit the application.

---

## **Example Commands**

1. **Webcam Input**:

   ```bash
   python webcam.py -o output_video.avi
   ```

2. **Video File Input**:

   ```bash
   python3 webcam.py -f input_video.mp4 -o output_video.avi
   ```

---

## **Test Output**

- **Real-Time Display**:
  - The application shows the video feed with detected objects and tracking annotations.
  - [Google drive link](#)
- **Processed Video**:
  - If the `o` flag is provided, the processed video will be saved with detection and tracking annotations.
  - [Google drive link](#)
