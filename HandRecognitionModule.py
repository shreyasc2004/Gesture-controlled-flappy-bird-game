import cv2
import mediapipe as mp
import numpy as np

class HandRecognizer:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        """
        Initializes the HandRecognizer.
        :param mode: Whether to treat the input images as a batch or a video stream
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum detection confidence threshold
        :param trackCon: Minimum tracking confidence threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        # Mediapipe's model complexity for hand landmarks (0 or 1)
        self.modelComplexity = 1
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mp_draw = mp.solutions.drawing_utils

    def get_landmarks(self, frame):
        """
        Finds hand landmarks in a frame and returns them as a flat list
        of 63 coordinates, NOW RELATIVE TO THE WRIST.
        
        :param frame: The video frame to process
        :return: A list of 63 relative landmarks (or an empty list) and the annotated frame
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        relative_landmarks = []

        if results.multi_hand_landmarks:
            # We only use the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on the frame for visualization
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # --- THIS IS THE CRITICAL FIX ---
            # Get the wrist (landmark 0) coordinates
            wrist_lm = hand_landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist_lm.x, wrist_lm.y, wrist_lm.z
            
            # Calculate all other landmarks *relative* to the wrist
            for lm in hand_landmarks.landmark:
                relative_landmarks.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])
            # --------------------------------

        # Return the 63 relative landmarks and the drawn frame
        return relative_landmarks, frame

    def __del__(self):
        # Release MediaPipe resources
        self.hands.close()

