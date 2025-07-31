import cv2

# Specify input and output paths
input_path = "/Users/ziyang/Downloads/test_video.mp4"
output_path = "/Users/ziyang/Downloads/resize_video.mp4"

# Define the desired width and height for resizing
new_width = 640
new_height = 360

# Open the input video
cap = cv2.VideoCapture(input_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Get the original video properties
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frame count

# Define the codec and create VideoWriter for output
# Use 'mp4v' codec, which is commonly compatible for .mp4 output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

# Process each frame
for _ in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Write the resized frame to the output video
    out.write(resized_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Resized video saved to", output_path)
