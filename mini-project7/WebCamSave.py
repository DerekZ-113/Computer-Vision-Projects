# USAGE: python WebCamSave.py -f video_file_name -o out_video.avi

# Import necessary packages
import cv2
import numpy as np
import time
import os
import argparse
import torch
from sklearn.preprocessing import LabelEncoder
from train import SimpleCNN  # Ensure train.py contains the model definition

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")
args = parser.parse_args()

# Load the trained model
model = SimpleCNN(num_classes=4)  # Adjust the number of classes if needed
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Define the label encoder
label_encoder = LabelEncoder().fit(["TV", "MobilePhone", "RemoteControl", "YourClass"])

# Set video source
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # Default camera

time.sleep(2.0)

# Get default resolutions
width = int(vs.get(3))
height = int(vs.get(4))

# Define codec and VideoWriter object
out_filename = args.out
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

# Function to perform classification on each frame
def classify_frame(frame):
    img = cv2.resize(frame, (64, 64))
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        label = label_encoder.inverse_transform([predicted.item()])[0]
    
    return label

# Loop over video frames
while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Perform classification and display the label on the frame
    label = classify_frame(frame)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)

    # Write the frame to the output video file (if specified)
    if args.out:
        out.write(frame)

    # Display the frame
    cv2.imshow("Video Classification", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break on 'q' key press
    if key == ord("q"):
        break

# Release video resources
vs.release()
out.release()
cv2.destroyAllWindows()
