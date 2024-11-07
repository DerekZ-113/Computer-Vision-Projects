# import the necessary packages
import cv2
import numpy as np
import time
import os
import argparse
import torch

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")
args = parser.parse_args()

# Load the trained YOLOv5 model(placeholder)
model_path = '/Users/placeholder/Downloads/best.pt'  # Path to your YOLOv5 model weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.conf = 0.25  # Set a custom confidence threshold

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

time.sleep(2.0)

# Get the default resolutions
width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
out_filename = args.out
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

while True:
    # grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

    # Convert frame to RGB (YOLOv5 expects RGB images)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float() / 255.0

    # Perform inference
    results = model(img)  # Ensure img is correctly preprocessed

    # Assuming results are tensors and not using .xyxy attribute
    # Convert results to CPU and then to numpy array if not done so already
    results = results[0].cpu().numpy()  # Assuming batch size of 1

    # Draw detection results on the image
    for detection in results:
        # Ensure detection has all expected values (x1, y1, x2, y2, confidence, class)
        if len(detection) >= 6:
            x1, y1, x2, y2, conf, cls = detection[:6]
            if conf > model.conf:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = f"{model.names[int(cls)]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("Unexpected format for detection:", detection)

    # Write the frame to the output video file
    if args.out:
        out.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release the video capture object
vs.release()
out.release()
cv2.destroyAllWindows()
