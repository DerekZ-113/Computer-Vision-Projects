# USAGE: python WebCamSave.py -f video_file_name -o out_video.avi
# USAGE: python3 WebCamSave.py -f lane_test1.mp4 -o out_lane_test1.avi
# USAGE: python3 WebCamSave.py -f lane_test2.mp4 -o out_lane_test2.avi
# USAGE: python3 WebCamSave.py -f lane_test3.mp4 -o out_lane_test3.avi


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
width  = int(vs.get(3))
height = int(vs.get(4))

# Define the codec and create a VideoWriter object
out_filename = args.out
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

def color_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 白色滤波范围
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    filtered_image = cv2.bitwise_and(image, image, mask=mask_white)
    return filtered_image

def region_of_interest(image):
    height, width = image.shape
    mask = np.zeros_like(image)
    polygon = np.array([[(0, height), (width, height), (width // 2, int(height * 0.6))]])
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if slope < -0.3:  # 左侧车道线阈值
                left_fit.append((slope, intercept))
            elif slope > 0.3:  # 右侧车道线阈值
                right_fit.append((slope, intercept))

    left_line = make_line_coordinates(image, np.mean(left_fit, axis=0) if left_fit else None)
    right_line = make_line_coordinates(image, np.mean(right_fit, axis=0) if right_fit else None)
    return left_line, right_line

def make_line_coordinates(image, line_parameters):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # 蓝色线条
    return line_image


# loop over the frames from the video stream
while True:
    # grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

    # Apply color filtering to isolate white lane lines
    filtered_image = color_filter(frame)

    # Convert the enhanced image to grayscale
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection to identify edges in the image
    edges = cv2.Canny(blurred, 50, 150)

    # Apply region of interest mask to focus only on the lanes area
    cropped_edges = region_of_interest(edges)

    # Perform Hough Line Transform to detect lines in the image
    lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi/180, threshold=50, 
                            minLineLength=40, maxLineGap=100)

    # Process the detected lines to find the left and right lane lines
    left_line, right_line = average_slope_intercept(frame, lines)

    # Draw the lane lines onto a blank image
    line_image = display_lines(frame, [left_line, right_line])

    # Combine the line image with the original frame to display detected lanes
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    if args.out:
        out.write(combined_image)

    cv2.imshow("Lane Detection", combined_image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

vs.release()
out.release()
cv2.destroyAllWindows()