import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('model_lunges.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

# Function to preprocess the input data
def preprocess_data(frame):
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            x_.append(x)
            y_.append(y)

        for landmark in results.pose_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))
    else:
        return None
    # Reshape the data to match the model's input shape
    data_aux = np.array(data_aux).reshape(1, -1)

    return data_aux

# Main function for real-time inference
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # Skip to next iteration if frame not captured

        # Preprocess the input data
        data_aux = preprocess_data(frame)

        if data_aux is not None:


            # Perform inference using the model
            prediction = model.predict(data_aux)

            # Determine the predicted label
            predicted_label = 'correct' if prediction[0] < 0.5 else 'incorrect'

            # Display the predicted label on the frame
            cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if predicted_label == 'correct' else (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('frame', frame)

        # Check for key press to exit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Entry point of the script
if __name__ == "__main__":
    main()