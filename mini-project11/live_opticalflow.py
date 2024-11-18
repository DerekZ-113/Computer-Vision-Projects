import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import time

# Load YOLO model (using a smaller model for better performance)
yolo_model = YOLO('yolov8n.pt')

# Initialize variables
object_tracks = {}  # Dictionary to store tracking info for each object
frame_count = 0
next_object_id = 0  # Unique ID generator for new objects
max_disappearance_frames = 10  # Max frames to keep a lost object

# Optical flow parameters (adjusted for performance)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,  # Reduced pyramid levels for faster computation
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Start video capture
cap = cv2.VideoCapture(0)

# Set desired frame width and height (resize for performance)
desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Read the first frame to initialize the mask
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Resize frame if necessary
frame = cv2.resize(frame, (desired_width, desired_height))

mask = np.zeros_like(frame)  # Initialize global mask

# Convert frame to grayscale
frame_gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Specify the classes to track (e.g., 'person', 'car')
classes_to_track = ['person']

# Initialize variables for FPS calculation
fps = 0
frame_counter = 0
start_time = time.time()

# Initialize tracking state
tracking_enabled = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame if necessary
    frame = cv2.resize(frame, (desired_width, desired_height))

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if tracking_enabled:
        # --- Tracking Code Starts Here ---

        if frame_count % 3 == 0:
            # Run YOLO detection every 3 frames
            results = yolo_model.predict(source=frame, stream=True)

            current_detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    label = yolo_model.names[int(box.cls)]  # Object class label
                    confidence = box.conf[0]  # Confidence score

                    if confidence > 0.5 and label in classes_to_track:
                        center_x = x1 + (x2 - x1) // 2
                        center_y = y1 + (y2 - y1) // 2
                        bbox = (x1, y1, x2, y2)
                        # Use the center of the bounding box as the feature point
                        feature_point = np.array([[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)
                        current_detections.append({
                            "label": label,
                            "confidence": confidence,
                            "center": (center_x, center_y),
                            "bbox": bbox,
                            "feature_point": feature_point,
                        })

            # Prepare cost matrix for matching
            track_ids = list(object_tracks.keys())
            detection_ids = list(range(len(current_detections)))
            cost_matrix = np.zeros((len(track_ids), len(detection_ids)), dtype=np.float32)

            for i, track_id in enumerate(track_ids):
                track_data = object_tracks[track_id]
                track_center = track_data["center"]
                for j, detection in enumerate(current_detections):
                    detection_center = detection["center"]
                    distance = np.linalg.norm(np.array(track_center) - np.array(detection_center))
                    cost_matrix[i, j] = distance

            # Solve the assignment problem
            if cost_matrix.size > 0:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            else:
                row_ind, col_ind = np.array([]), np.array([])

            # Update tracks with matched detections
            unmatched_tracks = set(track_ids)
            unmatched_detections = set(detection_ids)
            updated_tracks = {}

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 50:  # Match threshold
                    track_id = track_ids[r]
                    detection = current_detections[c]
                    updated_tracks[track_id] = {
                        "feature_point": detection["feature_point"],
                        "bbox": detection["bbox"],
                        "center": detection["center"],
                        "label": detection["label"],
                        "trajectory": object_tracks[track_id]["trajectory"] + [detection["center"]],
                        "disappearance_frames": 0,
                    }
                    unmatched_tracks.discard(track_ids[r])
                    unmatched_detections.discard(c)
                else:
                    unmatched_tracks.add(track_ids[r])
                    unmatched_detections.add(c)

            # Increment disappearance frames and remove lost tracks
            for track_id in unmatched_tracks:
                track_data = object_tracks[track_id]
                track_data["disappearance_frames"] += 1
                if track_data["disappearance_frames"] < max_disappearance_frames:
                    updated_tracks[track_id] = track_data

            # Create new tracks for unmatched detections
            for idx in unmatched_detections:
                detection = current_detections[idx]
                updated_tracks[f"{detection['label']}_{next_object_id}"] = {
                    "feature_point": detection["feature_point"],
                    "bbox": detection["bbox"],
                    "center": detection["center"],
                    "label": detection["label"],
                    "trajectory": [detection["center"]],
                    "disappearance_frames": 0,
                }
                next_object_id += 1

            # Update object_tracks with updated_tracks
            object_tracks = updated_tracks

        # Optical flow tracking for each object (runs every frame)
        for track_id, track_data in object_tracks.items():
            if len(track_data["feature_point"]) > 0:
                new_point, status, error = cv2.calcOpticalFlowPyrLK(
                    frame_gray_prev, frame_gray, track_data["feature_point"], None, **lk_params
                )

                if new_point is not None and status.sum() > 0:
                    good_new = new_point[status == 1]
                    good_old = track_data["feature_point"][status == 1]

                    # Update center position
                    if len(good_new) > 0:
                        track_data["center"] = tuple(good_new[0].astype(int).flatten())
                        track_data["trajectory"].append(track_data["center"])

                    # Draw tracking line
                    if len(track_data["trajectory"]) > 1:
                        for i in range(1, len(track_data["trajectory"])):
                            cv2.line(
                                mask,
                                track_data["trajectory"][i - 1],
                                track_data["trajectory"][i],
                                (0, 255, 0),
                                2,
                            )
                    frame = cv2.circle(frame, track_data["center"], 5, (0, 0, 255), -1)

                    # Update tracking data
                    track_data["feature_point"] = good_new.reshape(-1, 1, 2)
                else:
                    # If tracking fails, mark the track as lost
                    track_data["disappearance_frames"] += 1

        # Combine the global mask with the frame
        combined_frame = cv2.add(frame, mask)

        # Draw bounding boxes and labels
        for track_id, track_data in object_tracks.items():
            x1, y1, x2, y2 = track_data["bbox"]
            cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                combined_frame,
                track_id,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

    else:
        # If tracking is disabled, just display the frame
        combined_frame = frame.copy()

    # Calculate FPS
    frame_counter += 1
    if (time.time() - start_time) >= 1.0:
        fps = frame_counter / (time.time() - start_time)
        frame_counter = 0
        start_time = time.time()

    # Display the FPS on the frame
    cv2.putText(
        combined_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    # Display the frame
    cv2.imshow("Object Detection and Tracking", combined_frame)

    # Update previous frame and increment frame count
    frame_gray_prev = frame_gray.copy()
    frame_count += 1

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear tracking data
        object_tracks.clear()
        mask = np.zeros_like(frame)
        print("Tracking data cleared.")
    elif key == ord('t'):
        tracking_enabled = not tracking_enabled
        if tracking_enabled:
            print("Tracking enabled.")
        else:
            # Clear tracking data and mask when tracking is disabled
            object_tracks.clear()
            mask = np.zeros_like(frame)
            print("Tracking disabled and data cleared.")

# Release resources
cap.release()
cv2.destroyAllWindows()
