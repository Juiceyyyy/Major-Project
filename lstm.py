import os
import cv2
import numpy as np
from collections import deque
from keras.models import load_model
from keras.optimizers import Adam

# Load the LSTM model
model = load_model('lstm/human_detection_model.h5')  # Load your LSTM model

# Parameters for LSTM predictions
sequence_length = 10  # Number of frames to consider for predictions

# Set up tracking trajectories
tracking_trajectories = {}

def predict(frame_sequence):
    """Predict the presence of a person in a sequence of frames using the LSTM model."""
    # Reshape and normalize the input for the LSTM model
    input_data = np.array(frame_sequence).reshape((1, sequence_length, -1)) / 255.0
    prediction = model.predict(input_data)
    return 1 if prediction[0][0] > 0.2 else 0  # Binary classification threshold

def process_frame(frame, frame_sequence, frameId, track=True):
    """Process the frame and update tracking information."""
    global input_video_name
    bboxes = []

    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    labels_file_path = os.path.abspath(f'./output/{input_video_name}_labels.txt')

    with open(labels_file_path, 'a') as file:
        if track:
            # Resize frame for LSTM input
            resized_frame = cv2.resize(frame, (32, 32))
            frame_sequence.append(resized_frame)

            # Log frame sequence size
            print(f"Current sequence length: {len(frame_sequence)}")

            if len(frame_sequence) >= sequence_length:
                presence = predict(list(frame_sequence))
                
                bboxes.append([0, 0, frame.shape[1], frame.shape[0], presence])
                
                if presence == 1:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

                # Log presence detected
                print(f"Frame ID: {frameId}, Presence Detected: {presence}")

        for item in bboxes:
            bbox_coords = item[:4]
            presence = item[4]
            line = f'{frameId} {presence} {bbox_coords[0]} {bbox_coords[1]} {bbox_coords[2]} {bbox_coords[3]} -1 -1 -1 -1\n'
            file.write(line)

    return frame


def load_dataset(file_path):
    """Load the dataset from a .npz file."""
    data = np.load(file_path)
    X_train = data['X_train']  # Use the key that stores your training features
    y_train = data['y_train']   # Use the key that stores your training labels
    return X_train, y_train

def compile_and_train_model(X_train, y_train):
    """Compile and train the LSTM model."""
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=32)

def process_video(args):
    """Process the video for person detection and tracking."""
    print(args)
    source = args['source']
    track_ = args['track']
    count_ = args['count']

    global input_video_name
    cap = cv2.VideoCapture(int(source) if source == '0' else source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
    input_video_name = os.path.splitext(os.path.basename(source))[0]
    out = cv2.VideoWriter(f'output/{input_video_name}_output.mp4', fourcc, fps, (frame_width, frame_height))

    if not cap.isOpened():
        print(f"Error: Could not open video file {source}.")
        return

    frameId = 0
    frame_sequence = deque(maxlen=sequence_length)  # Maintain a rolling window of frames

    while True:
        frameId += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame and pass frameId
        frame = process_frame(frame, frame_sequence, frameId, track_)

        out.write(frame)  # Write the processed frame to output

    cap.release()
    out.release()
    print(f'Output video saved as: output/{input_video_name}_output.mp4')

if __name__ == "__main__":
    # Load dataset
    dataset_path = 'dataset/personpath22/processed_data.npz'
    X_train, y_train = load_dataset(dataset_path)

    # Compile and train the model
    compile_and_train_model(X_train, y_train)

    # Here you would parse the command line arguments (args) as needed
    args = {
        'source': 'sample.mp4',
        'track': True,
        'count': True
    }
    process_video(args)
