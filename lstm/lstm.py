import os
import cv2
import numpy as np
import time
from collections import deque, Counter
import tensorflow as tf

# Load the LSTM model
model = tf.load_model('human_detection_model.keras')

# Initialize tracking variables
tracking_trajectories = {}
colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')

def process_frame_sequence(frame):
    """Process a frame sequence for LSTM input."""
    input_frame = cv2.resize(frame, (64, 64))  # Adjust based on your model's requirements
    input_data = input_frame / 255.0  # Normalize to [0, 1]
    input_data = input_data.reshape(1, 1, -1)  # Shape: (1, 1, 12288)

    predictions = model.predict(input_data)  # Output shape: (1, 1)

    print(f"Predictions shape: {predictions.shape}")  # Debug the output shape
    print(f"Predictions: {predictions}")  # Debug the output structure

    confidence_score = predictions[0][0]  # Access the single prediction
    print(f"Confidence score: {confidence_score}")

    confidence_threshold = 0.2  # Set your confidence threshold
    if confidence_score >= confidence_threshold:
        # Define bounding box coordinates (for visualization purposes)
        h, w, _ = frame.shape
        xmin, ymin, xmax, ymax = int(w/4), int(h/4), int(3*w/4), int(3*h/4)
        
        # Draw bounding box and score on the frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(frame, f'Score: {confidence_score:.2f}', (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return [(xmin, ymin, xmax, ymax, 0, confidence_score)]  # Return bounding box with dummy class ID
    else:
        print(f"Ignoring prediction with low score: {confidence_score}")
        return []  # No valid predictions

def draw_bbox_and_track(frame, bboxes):
    """Draw bounding boxes and track objects."""
    for bbox in bboxes:
        xmin, ymin, xmax, ymax, obj_class, score = bbox

        # Draw bounding box
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 225), 2)

        # Track centroid of the object
        centroid_x = (xmin + xmax) / 2
        centroid_y = (ymin + ymax) / 2
        object_id = int(obj_class)  # Use class as ID for simplicity

        if object_id not in tracking_trajectories:
            tracking_trajectories[object_id] = deque(maxlen=5)
        tracking_trajectories[object_id].append((centroid_x, centroid_y))

        for i in range(1, len(tracking_trajectories[object_id])):
            cv2.line(
                frame,
                (int(tracking_trajectories[object_id][i - 1][0]),
                 int(tracking_trajectories[object_id][i - 1][1])),
                (int(tracking_trajectories[object_id][i][0]),
                 int(tracking_trajectories[object_id][i][1])),
                (255, 255, 255), 2
            )

def process_video(args):
    """Process video to track and count objects."""
    source = args['source']
    track_ = args['track']
    count_ = args['count']
    input_video_name = os.path.splitext(os.path.basename(source))[0]

    cap = cv2.VideoCapture(int(source) if source == '0' else source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(f'output/{input_video_name}_output.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), 15,
                          (frame_width, frame_height))

    if not cap.isOpened():
        print(f"Error: Could not open video file {source}.")
        return

    frameId = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes = process_frame_sequence(frame)
        draw_bbox_and_track(frame, bboxes)

        if track_ and count_:
            counts = Counter([bbox[4] for bbox in bboxes])  # Using the class ID for counting
            display_count = f"Counts: {dict(counts)}"
            cv2.putText(frame, display_count, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        if frameId % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = frameId / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        cv2.imshow(f"LSTM_{source}", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frameId += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process video with LSTM.')
    parser.add_argument('--source', type=str, default='0', help='Input video file path or camera index')
    parser.add_argument('--track', action='store_true', help='Track objects')
    parser.add_argument('--count', action='store_true', help='Count objects')

    args = parser.parse_args()
    process_video({'source': args.source, 'track': args.track, 'count': args.count})
