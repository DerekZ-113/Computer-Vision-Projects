import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time

# Argument parser for video input/output
parser = argparse.ArgumentParser(description="YOLO Object Detection and Optical Flow")
parser.add_argument("-f", "--file", type=str, help="Path to the video file (leave empty for webcam)")
parser.add_argument("-o", "--out", type=str, help="Output video file name")
args = parser.parse_args()

# Load YOLOv5 pre-trained model
# yolov5s.pt: Small (fastest, least accurate)
# yolov5m.pt: Medium
# yolov5l.pt: Large
# yolov5x.pt: Extra-large (most accurate, slowest)
yolo_model = YOLO('yolov5s.pt')

# Initialize video source
if args.file:
    cap = cv2.VideoCapture(args.file)
else:
    cap = cv2.VideoCapture(0)  # 0 for default webcam

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up video writer if output is specified
if args.out:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height), True)

# Initialize variables
roi_selected = False
old_points = None
mask = None
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

old_gray = None

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot grab frame.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- YOLO Object Detection ----
    results = yolo_model.predict(source=frame, stream=True)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo_model.names[int(box.cls)]
            score = box.conf[0]
            # Draw detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ---- Optical Flow Tracking ----
    if roi_selected:
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

        if new_points is not None:
            good_new = new_points[status == 1]
            good_old = old_points[status == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = int(a), int(b), int(c), int(d)
                mask = cv2.line(mask, (a, b), (c, d), color=(0, 255, 0), thickness=2)
                frame = cv2.circle(frame, (a, b), 5, color=(0, 0, 255), thickness=-1)

            img = cv2.add(frame, mask)
            cv2.imshow("Object Tracking", img)

            # Update old points and frame
            old_points = good_new.reshape(-1, 1, 2)
            old_gray = frame_gray.copy()
        else:
            print("Tracking lost.")
            roi_selected = False
    else:
        cv2.imshow("Object Tracking", frame)

    # Wait for user input
    key = cv2.waitKey(30) & 0xFF

    if key == ord('s') and not roi_selected:
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi
        old_points = np.array([[x + w // 2, y + h // 2]], dtype=np.float32).reshape(-1, 1, 2)
        cv2.destroyWindow("Select ROI")
        mask = np.zeros_like(frame)
        old_gray = frame_gray.copy()
        roi_selected = True

    if key == ord('q'):
        break

    # Write output frame
    if args.out:
        out.write(frame)

# Release resources
cap.release()
if args.out:
    out.release()
cv2.destroyAllWindows()

#python3 webcam.py -f test.mp4 -o output_video.avi