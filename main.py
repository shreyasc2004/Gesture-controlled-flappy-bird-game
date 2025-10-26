import cv2
import pygame
from HandRecognitionModule import HandRecognizer
from GestureInterpreterModule import GestureInterpreter
from GameEnvironmentModule import FlappyBirdGame
import sys # Import sys for exiting

# --- Initialization ---
print("Initializing modules...")
try:
    # 1. Initialize OpenCV Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    # 2. Initialize MediaPipe Hand Recognizer
    # Uses the new 'get_landmarks' method
    recognizer = HandRecognizer()

    # 3. Initialize Gesture Interpreter (The MLP Model)
    # This will load gesture_model.keras, scaler.pkl, etc.
    interpreter = GestureInterpreter()

    # 4. Initialize Pygame Game Environment
    game = FlappyBirdGame()

except Exception as e:
    print(f"Error during initialization: {e}")
    print("Please ensure all model files (keras, pkl) exist.")
    print("Run DataCollector.py and ModelTrainer.py first.")
    sys.exit()

print("Initialization complete. Starting game...")

# --- Main Game Loop ---
while True:
    # 1. ** THIS IS THE FIX **
    # We removed the separate game.handle_events() call,
    # as it's now handled inside game.update()
    
    # 2. Capture Frame from Webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break
    
    # Flip frame horizontally for a "mirror" view
    frame = cv2.flip(frame, 1)

    # 3. Get hand landmarks from the recognizer
    # 'landmarks' is the flat list of 63 relative coordinates
    # 'annotated_frame' is the image with the hand skeleton drawn on it
    landmarks, annotated_frame = recognizer.get_landmarks(frame.copy())

    # 4. Classify Gesture
    # Feed the 63 landmarks into the MLP model
    current_gesture = interpreter.classify(landmarks)

    # 5. Update Game State
    # Tell the game logic what gesture is being made
    # This method ALSO handles pygame events (like closing the window)
    game.update(current_gesture)

    # 6. ** THIS IS THE NEW CHECK **
    # Check if game.update() detected a QUIT event
    if not game.running:
        print("Pygame window closed. Exiting.")
        break

    # 7. Draw Everything
    # Draw the Flappy Bird game elements (bird, pipes, score)
    game.draw()
    
    # 8. Display the Webcam Feed
    # Show the OpenCV window with the hand skeleton and current gesture
    cv2.putText(annotated_frame, f"Gesture: {current_gesture}", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Gesture Control', annotated_frame)

    # 9. Check for Quit Key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("'q' pressed. Exiting.")
        break

# --- Cleanup ---
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
game.quit()

