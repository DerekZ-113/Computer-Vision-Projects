# Import necessary packages
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

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

# Image stitching to create the panorama
def stitch_images(frames, max_width=8000):
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
