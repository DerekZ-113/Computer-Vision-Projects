# import the necessary packages
import cv2
import numpy as np
import time
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")

args = parser.parse_args()

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

time.sleep(2.0)

# Get the default resolutions
width = int(vs.get(3))
height = int(vs.get(4))

# Define the codec and create a VideoWriter object
out_filename = args.out
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

# Function to apply a color filter to isolate white and yellow lane lines
def color_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for white color in HSV
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 25, 255])

    # Define range for yellow color in HSV
    yellow_lower = np.array([18, 94, 140])
    yellow_upper = np.array([48, 255, 255])

    # Create masks for white and yellow colors
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Combine both masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    filtered_image = cv2.bitwise_and(image, image, mask=combined_mask)
    return filtered_image

# Define the region of interest (ROI) function
def region_of_interest(image):
    mask = np.zeros_like(image)
    height, width = image.shape
    polygon = np.array([[(0, height), (width, height), (int(width * 0.5), int(height * 0.6))]])
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Function to display lane lines on the frame
def display_lines(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return line_image

# loop over the frames from the video stream
while True:
    # grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

    # Apply color filter to the frame to isolate lane colors
    filtered_image = color_filter(frame)

    # Convert the filtered frame to grayscale
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply region of interest mask
    cropped_edges = region_of_interest(edges)

    # Perform Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi/180, threshold=50, 
                            minLineLength=40, maxLineGap=100)

    # Create a line image and add it to the original frame
    line_image = display_lines(frame, lines)
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # Write the frame to the output video file
    if args.out:
        out.write(combined_image)

    # Show the output frame
    cv2.imshow("Lane Detection", combined_image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release the video capture object
vs.release()
out.release()
cv2.destroyAllWindows()
