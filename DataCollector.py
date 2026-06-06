import cv2                     # OpenCV - webcam and image processing
import csv                     # Save data into CSV file
import os                      # File handling
from HandRecognitionModule import HandRecognizer  # Custom MediaPipe wrapper

# ---------------- CONFIGURATION ----------------

DATA_PATH = 'gestures.csv'     # Dataset file

# Gesture classes to collect
CLASSES = [
    'NEUTRAL',
    'THUMBS_UP',
    'THUMBS_DOWN'
]

# Number of samples per gesture
SAMPLES_PER_GESTURE = 200


# ---------------- INITIALIZATION ----------------

# Create hand landmark detector
recognizer = HandRecognizer()

# Open webcam
cap = cv2.VideoCapture(0)

# Check webcam availability
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# ---------------- CREATE CSV HEADER ----------------

# Create CSV only if it doesn't already exist
if not os.path.exists(DATA_PATH):

    with open(DATA_PATH, mode='w', newline='') as f:

        writer = csv.writer(f)

        # First column = gesture label
        header = ['gesture']

        # MediaPipe gives 21 landmarks
        # Each landmark has x,y,z coordinates
        for i in range(21):
            header += [
                f'lm_{i}_x',
                f'lm_{i}_y',
                f'lm_{i}_z'
            ]

        # Write header row
        writer.writerow(header)

print(f"Data will be saved to: {DATA_PATH}")


# ---------------- DATA COLLECTION FUNCTION ----------------

def collect_data():

    # Loop through each gesture class
    for gesture_name in CLASSES:

        print(f"\nGet ready to collect data for: {gesture_name}")
        print("Press 's' to start collecting. Press 'q' to quit.")

        # -------- WAIT FOR USER TO PRESS S --------

        while True:

            # Capture frame from webcam
            ret, frame = cap.read()

            if not ret:
                print("Error reading frame")
                return

            # Mirror image for natural interaction
            frame = cv2.flip(frame, 1)

            # Detect hand landmarks
            landmarks, frame_with_hands = \
                recognizer.get_landmarks(frame.copy())

            # Show instructions on screen
            cv2.putText(
                frame_with_hands,
                f"Press 's' to start collecting {gesture_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                'Data Collector',
                frame_with_hands
            )

            key = cv2.waitKey(1) & 0xFF

            # Start collection
            if key == ord('s'):
                break

            # Quit program
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # -------- COLLECT DATA --------

        print(
            f"Collecting {SAMPLES_PER_GESTURE} "
            f"samples for {gesture_name}"
        )

        sample_count = 0

        # Continue until 200 samples collected
        while sample_count < SAMPLES_PER_GESTURE:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Detect landmarks
            landmarks, frame_with_hands = \
                recognizer.get_landmarks(frame.copy())

            # Save only if a hand is detected
            if landmarks and len(landmarks) == 63:

                # MediaPipe returns:
                # 21 landmarks × 3 coordinates
                # = 63 values

                lm_list = landmarks

                # Create row:
                # [Gesture Label + Landmark Values]
                row_data = [gesture_name] + lm_list

                # Append row to CSV
                with open(
                    DATA_PATH,
                    mode='a',
                    newline=''
                ) as f:

                    writer = csv.writer(f)

                    writer.writerow(row_data)

                # Increment sample count
                sample_count += 1

                # Calculate progress percentage
                progress = int(
                    (sample_count /
                     SAMPLES_PER_GESTURE) * 100
                )

                # Display progress
                cv2.putText(
                    frame_with_hands,
                    f"Collecting {gesture_name}: {progress}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # Show webcam feed
            cv2.imshow(
                'Data Collector',
                frame_with_hands
            )

            # Quit current collection
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Collection interrupted")
                break

    # -------- CLEANUP --------

    print("Data collection complete!")

    cap.release()

    cv2.destroyAllWindows()


# ---------------- PROGRAM ENTRY ----------------

if __name__ == "__main__":

    # Start dataset collection
    collect_data()
