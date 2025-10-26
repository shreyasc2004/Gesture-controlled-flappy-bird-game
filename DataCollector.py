import cv2
import csv
import os
from HandRecognitionModule import HandRecognizer # Uses the file in your canvas

# --- CONFIGURATION ---
DATA_PATH = 'gestures.csv'
CLASSES = ['NEUTRAL', 'THUMBS_UP', 'THUMBS_DOWN']
SAMPLES_PER_GESTURE = 200 # Collect 200 samples for each gesture

# --- Setup ---
recognizer = HandRecognizer()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# File header
if not os.path.exists(DATA_PATH):
    with open(DATA_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['gesture']
        for i in range(21): # 21 landmarks
            header += [f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z']
        writer.writerow(header)
print(f"Data will be saved to: {DATA_PATH}")

def collect_data():
    """Main loop to collect data for each gesture."""
    
    for gesture_name in CLASSES:
        print(f"\nGet ready to collect data for: {gesture_name}")
        print("Press 's' to start collecting. Press 'q' to quit.")
        
        # Wait for user to press 's'
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                return
            frame = cv2.flip(frame, 1)
            
            # ** THIS IS THE FIX **
            # We call get_landmarks, which returns the landmarks AND the drawn frame
            landmarks, frame_with_hands = recognizer.get_landmarks(frame.copy())
            
            cv2.putText(frame_with_hands, f"Press 's' to start collecting {gesture_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Data Collector', frame_with_hands)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            if key == ord('q'):
                print("Quitting...")
                cap.release()
                cv2.destroyAllWindows()
                return

        # Collect samples
        print(f"Collecting {SAMPLES_PER_GESTURE} samples for {gesture_name}...")
        sample_count = 0
        while sample_count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # ** THIS IS THE FIX **
            landmarks, frame_with_hands = recognizer.get_landmarks(frame.copy())
            
            # Only save data if a hand was actually detected
            if landmarks and len(landmarks) == 63:
                # ** THIS IS THE FIX **
                # The 'landmarks' variable is already the flat list we need
                lm_list = landmarks 
                
                # Append gesture name to the start of the list
                row_data = [gesture_name] + lm_list
                
                # Save to CSV
                with open(DATA_PATH, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)
                    
                sample_count += 1
                
                # Display progress
                progress = int((sample_count / SAMPLES_PER_GESTURE) * 100)
                cv2.putText(frame_with_hands, f"Collecting {gesture_name}: {progress}%",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Data Collector', frame_with_hands)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Collection interrupted.")
                break # Move to next gesture
    
    print("\nData collection complete!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()

