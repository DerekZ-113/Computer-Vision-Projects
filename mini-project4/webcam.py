import cv2

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

orb = cv2.ORB_create()
apply_orb = False

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
    
    # Stack the frames from both cameras horizontally
    combined = cv2.hconcat([frame1, frame2])

    # ORB
    if apply_orb:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        combined_gray = cv2.hconcat([gray1, gray2])
    
        keypoints, descriptors = orb.detectAndCompute(combined_gray, None)
        combined = cv2.drawKeypoints(combined, keypoints, None, color=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)


    # Display the resulting frames
    cv2.imshow('Camera 1 and Camera 2', combined)

    # Break the loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('o'):
        apply_orb = not apply_orb
    if key == ord('q'):
        break

# Release the cameras and close all OpenCV windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
