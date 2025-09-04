#!/usr/bin/env python3
"""
Test the fixed camera predictor with MediaPipe hand detection
"""

import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_hand_detection_only():
    """Test just the hand detection part without models"""
    import mediapipe as mp
    
    print("Testing MediaPipe hand detection...")
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
    
    # Start camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False
    
    print("Hand detection test started - Press ESC to quit")
    print("You should see:")
    print("- Green landmarks on your hand")
    print("- Blue bounding box around your hand")
    print("- Connection lines between landmarks")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style()
                )
                
                # Get bounding box from landmarks
                h, w, c = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
                
                # Add margin
                margin = 20
                x1 = max(0, x_min - margin)
                y1 = max(0, y_min - margin)
                x2 = min(w, x_max + margin)
                y2 = min(h, y_max + margin)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "HAND DETECTED", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.putText(frame, "SUCCESS: Hand detection working!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hands detected", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, "Press ESC to quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Fixed Hand Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("Hand detection test completed successfully!")
    return True

if __name__ == "__main__":
    test_hand_detection_only()