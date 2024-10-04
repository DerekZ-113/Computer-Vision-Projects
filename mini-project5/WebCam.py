# USAGE: python CamTest.py

# import the necessary packages
import cv2
import time
import os

# Calculate FPS averaged over the given time interval (default: 1 second).
def calculate_fps(frame_count, prev_time, interval=1.0):
    current_time = time.time()
    elapsed_time = current_time - prev_time

    if elapsed_time >= interval:
        fps = frame_count / elapsed_time
        prev_time = current_time
        frame_count = 0  # Reset frame count after each interval
        return fps, prev_time, frame_count
    else:
        return None, prev_time, frame_count
    
def main():
    # Open Video Camera
    vs = cv2.VideoCapture(0)  # 0 is the default camera
    time.sleep(2.0)

    # Initialize FPS variables
    prev_time = time.time()
    frame_count = 0
    fps = 0

    # loop over the frames from the video stream
    while True:
        # grab the frame from video stream
        ret, frame = vs.read()

        # Add your code HERE: For example,
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 165, 255,
                        cv2.THRESH_BINARY)[1]


        # show the output frame
        cv2.imshow("Frame", thresh)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

# Run the main function
if __name__ == "__main__":
    main()