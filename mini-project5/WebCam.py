# Import necessary packages
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# Calculate FPS averaged over the given time interval (default: 1 second).
def calculate_fps(frame_count, prev_time, interval=1.0):
    '''
    Args:
    - frame_count: current count of processed frames
    - prev_time: the last time the FPS calculation was done
    - interval: the time interval over which the FPS is calculated (default is 1 second)

    This function computes the frames per second (FPS) over a given time interval.
    It tracks how many frames have been processed and the time that has passed.

    By checking the elapsed time between the previous time and current time,
    the function calculates the FPS if the interval has passed and resets the frame count and previous time.
    If the interval hasn't passed, it returns None and waits for the next frame.

    Returns:
    - fps: the calculated frames per second
    - prev_time: the updated previous time
    - frame_count: reset or updated frame count
    '''
    current_time = time.time()
    elapsed_time = current_time - prev_time

    if elapsed_time >= interval:
        fps = frame_count / elapsed_time
        prev_time = current_time
        frame_count = 0  # Reset frame count after each interval
        return fps, prev_time, frame_count
    else:
        return None, prev_time, frame_count

# Image stitching to create the panorama
def stitch_images(frames, max_width=8000):
    '''
    Args:
    - frames: list of frames captured from the webcam
    - max_width: the maximum allowed width for the stitched image (default is 8000 pixels)

    This function stitches multiple frames into a single panorama using ORB for feature detection
    and BFMatcher for feature matching. It aligns consecutive frames and combines them into one wide image.

    ORB keypoints and descriptors are computed for each frame. BFMatcher matches keypoints between frames,
    and homography is used to align the frames. The frames are then warped and combined to create the panorama.

    Returns:
    - stitched_image: the final stitched panorama
    '''
    stitched_image = frames[0]
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(1, len(frames)):
        gray1 = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints and descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        # Match features using the BFMatcher
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract the matched keypoints
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for j, match in enumerate(matches):
            points1[j, :] = keypoints1[match.queryIdx].pt
            points2[j, :] = keypoints2[match.trainIdx].pt

        # Compute homography matrix using RANSAC
        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
        height, width = frames[i].shape[:2]

        # Limit the output width to avoid OpenCV errors
        # errors: (-215:Assertion failed) dst.cols < SHRT_MAX && dst.rows < SHRT_MAX && src.cols < SHRT_MAX && src.rows < SHRT_MAX in function 'remap'
        result_width = min(stitched_image.shape[1] + width, max_width)
        stitched_image = cv2.warpPerspective(stitched_image, H, (result_width, stitched_image.shape[0]))
        stitched_image[0:height, 0:width] = frames[i]

    return stitched_image

# Save panorama to disk

def save_panorama(panorama):
    '''
    Args:
    - panorama: the stitched image that will be saved

    This function saves the generated panorama to disk and displays it using OpenCV.

    Returns:
    None
    '''
    if panorama is not None:
        cv2.imshow("Panorama", panorama)
        save_path = "panorama.jpg"
        cv2.imwrite(save_path, panorama)
        print(f"Panorama saved to {save_path}")

def main():
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)
    
    capturing = False
    captured_frames = []
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        # Calculate and display FPS
        fps_text, prev_time, frame_count = calculate_fps(frame_count + 1, prev_time)
        if fps_text is not None:
            fps = fps_text
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Webcam Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # Start capturing frames when 's' is pressed
        if key == ord("s"):
            capturing = True
            print("Started capturing frames for panorama.")

        # Capture frames if capturing is active
        if capturing:
            captured_frames.append(frame.copy())
            
        # Stop capturing when 'a' is pressed
        if key == ord("a"):
            capturing = False
            print("Stopped capturing frames.")
            break

        # Exit on 'q' press
        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()
    
    # Stitch the captured frames to create panorama
    if len(captured_frames) > 1:
        print("Stitching frames to create panorama...")
        panorama = stitch_images(captured_frames)
        save_panorama(panorama)
        cv2.waitKey(0)
    else:
        print("Not enough frames to create a panorama.")

# Run the main function
if __name__ == "__main__":
    main()
