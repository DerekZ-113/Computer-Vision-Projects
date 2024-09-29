import cv2
import time
import numpy as np

def draw_matches_with_scores(img1, kp1, img2, kp2, matches, max_matches=10):
    """
    Draws the matches between two images with their scores.

    Parameters:
    - img1: First image.
    - kp1: Keypoints of the first image.
    - img2: Second image.
    - kp2: Keypoints of the second image.
    - matches: List of DMatch objects containing the matched keypoints.
    - max_matches: Maximum number of matches to display.

    Returns:
    - img_match: Image with matches drawn.
    """

    # Sort the matches based on their distance (score)
    matches = sorted(matches, key=lambda x: x.distance)

    # Select the top 'max_matches' matches
    display_matches = matches[:max_matches]

    # Draw matches with scores
    img_match = cv2.drawMatches(
        img1, kp1, img2, kp2, display_matches, None,
        matchColor=(0, 255, 0),  # Color of the matching lines
        singlePointColor=(255, 0, 0),  # Color of the keypoints
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Add the score (distance) as text on the matches
    for i, match in enumerate(display_matches):
        pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        score_text = f"Score: {match.distance:.2f}"

        # Put text on the matching line midpoint
        mid_point = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
        cv2.putText(img_match, score_text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return img_match

def main():
    # Initialize the cameras
    # Since we are using internal camera and external camera: they don't share the same resolution
    # Thus, we are adding steps (resize) to resolve frame1 and frame2 do not have the same number of rows
    cap1 = cv2.VideoCapture(0)  # First camera
    cap2 = cv2.VideoCapture(1)  # Second camera, change the index if necessary

    # Get the resolution of each camera
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"Camera 1 resolution: {width1}x{height1}")

    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"Camera 2 resolution: {width2}x{height2}")

    # Decide on the target resolution, in this case we are using the higher resolution of both cameras
    target_width = max(width1, width2)
    target_height = max(height1, height2)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Initialize FPS variables
    prev_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        # Capture frame-by-frame from the first camera
        ret1, frame1 = cap1.read()
        # Capture frame-by-frame from the second camera
        ret2, frame2 = cap2.read()

        # Check if frames are captured
        if not ret1 or not ret2:
            break

        # Resize both frames to the target resolution
        frame1 = cv2.resize(frame1, (target_width, target_height))
        frame2 = cv2.resize(frame2, (target_width, target_height))

        # Ensure both frames are of the same data type
        # if frame1.dtype != frame2.dtype:
        #     frame2 = frame2.astype(frame1.dtype)
        
        # # Stack the frames from both cameras horizontally
        # combined = cv2.hconcat([frame1, frame2])

        # ORB
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # combined_gray = cv2.hconcat([gray1, gray2])

        # Detect keypoints and compute descriptors using ORB
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
        
        # keypoints, descriptors = orb.detectAndCompute(combined_gray, None)
        # combined = cv2.drawKeypoints(combined, keypoints, None, color=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)

        # Create BFMatcher object with NORM_HAMMING for binary descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors using FLANN matcher
        if descriptors1 is not None and descriptors2 is not None:
            matches = bf.match(descriptors1, descriptors2)
            # Draw matches with scores
            matched_img = draw_matches_with_scores(frame1, keypoints1, frame2, keypoints2, matches, max_matches=10)

            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - prev_time

            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                prev_time = current_time
                frame_count = 0

            # Display FPS on the image
            cv2.putText(matched_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the matched image
            cv2.imshow('Matches with Scores', matched_img)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the cameras and close all OpenCV windows
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()