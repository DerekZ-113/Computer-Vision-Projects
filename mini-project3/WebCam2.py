import cv2
import numpy as np
import time

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use index 0 for default camera

# Initialize transformation mode
mode = 'gray'

def warp_image(frame, mode='gray'):
    rows, cols = frame.shape[:2]
    
    # Translation: Shift the image horizontally and/or vertically
    if mode == 'translation':
        # Shift right by 100 pixels and down by 50 pixels
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        transformed = cv2.warpAffine(frame, M, (cols, rows))
        
    # Rotation: Rotate the image around a specified point.
    elif mode == 'rotation':
        # Rotate the image 45 degrees
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        transformed = cv2.warpAffine(frame, M, (cols, rows))
        
    # Scaling: Resize the image by scaling up or down.
    elif mode == 'scaling':
        # Scale by a factor of 1.5
        scaled = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        new_rows, new_cols = scaled.shape[:2]
        start_x = (new_cols - cols) // 2
        start_y = (new_rows - rows) // 2
        transformed = scaled[start_y:start_y + rows, start_x:start_x + cols]
        
    # Perspective Transformation: Apply a perspective warp to simulate a change in viewpoint.
    elif mode == 'perspective':
        # Perspective Transform: Warp to simulate a different viewpoint
        pts1 = np.float32([[50, 50], [cols - 50, 50], [50, rows - 50], [cols - 50, rows - 50]])
        pts2 = np.float32([[10, 100], [cols - 10, 50], [100, rows - 100], [cols - 100, rows - 50]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(frame, M, (cols, rows))
        
    # Default mode: Convert to grayscale
    else:
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transformed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return transformed

prev_time = time.time()
frame_count = 0
fps = 0  # Initialize FPS variable

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    transformed_frame = warp_image(frame, mode)
    
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - prev_time
    
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        prev_time = current_time
        frame_count = 0
    
    # Display FPS on the image
    cv2.putText(transformed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Stack the original and transformed images horizontally
    combined_frame = cv2.hconcat([frame, transformed_frame])
    
    # Display the resulting frame
    cv2.imshow('Original and Transformed', combined_frame)
    
    # Read key for changing the transformation mode
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('t'):
        mode = 'translation'
    elif key == ord('r'):
        mode = 'rotation'
    elif key == ord('s'):
        mode = 'scaling'
    elif key == ord('p'):
        mode = 'perspective'

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
