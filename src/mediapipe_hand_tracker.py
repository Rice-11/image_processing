import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import json
import os
from datetime import datetime

class MediaPipeHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Focus on one hand for better performance
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # For gesture recording
        self.is_recording = False
        self.gesture_buffer = deque(maxlen=60)  # 2 seconds at 30 FPS
        self.landmarks_history = []
        
        # Data collection
        self.data_dir = "gesture_data"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/static", exist_ok=True)
        os.makedirs(f"{self.data_dir}/dynamic", exist_ok=True)
    
    def extract_landmarks(self, hand_landmarks):
        """Extract normalized hand landmarks as feature vector"""
        landmarks = []
        
        if hand_landmarks:
            # Get wrist position for normalization
            wrist = hand_landmarks.landmark[0]
            
            # Extract all 21 landmarks relative to wrist
            for landmark in hand_landmarks.landmark:
                landmarks.extend([
                    landmark.x - wrist.x,  # Relative x
                    landmark.y - wrist.y,  # Relative y  
                    landmark.z - wrist.z   # Relative z
                ])
        else:
            # Return zeros if no hand detected
            landmarks = [0.0] * 63  # 21 landmarks × 3 coordinates
            
        return np.array(landmarks)
    
    def detect_hand_movement(self, current_landmarks, window_size=10):
        """Detect if hand is moving based on landmark history"""
        if len(self.landmarks_history) < window_size:
            return False
            
        # Calculate movement by comparing recent landmarks
        recent_landmarks = np.array(self.landmarks_history[-window_size:])
        movement = np.std(recent_landmarks, axis=0)
        
        # If average movement across all landmarks exceeds threshold
        avg_movement = np.mean(movement)
        return avg_movement > 0.01  # Adjust threshold as needed
    
    def process_frame(self, frame):
        """Process frame and return landmarks + annotated frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks = None
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmarks
                landmarks = self.extract_landmarks(hand_landmarks)
                
                # Add to history for movement detection
                self.landmarks_history.append(landmarks)
                if len(self.landmarks_history) > 30:  # Keep last 30 frames
                    self.landmarks_history.pop(0)
        
        return landmarks, annotated_frame
    
    def start_static_recording(self, label):
        """Start recording static gesture"""
        self.current_label = label
        self.current_type = "static"
        self.static_samples = []
        print(f"Recording static gesture: {label}")
        print("Hold the pose steady. Press SPACE to capture, ESC to stop")
    
    def start_dynamic_recording(self, label):
        """Start recording dynamic gesture"""
        self.current_label = label
        self.current_type = "dynamic"
        self.gesture_buffer.clear()
        self.is_recording = True
        print(f"Recording dynamic gesture: {label}")
        print("Perform the gesture. Recording will auto-stop after 2 seconds of movement")
    
    def capture_static_sample(self, landmarks):
        """Capture a single static gesture sample"""
        if landmarks is not None:
            self.static_samples.append(landmarks.tolist())
            print(f"Captured sample {len(self.static_samples)} for {self.current_label}")
            return True
        return False
    
    def save_static_data(self):
        """Save collected static gesture data"""
        if hasattr(self, 'static_samples') and self.static_samples:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_dir}/static/{self.current_label}_{timestamp}.json"
            
            data = {
                "label": self.current_label,
                "type": "static",
                "samples": self.static_samples,
                "num_samples": len(self.static_samples),
                "timestamp": timestamp
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f)
            
            print(f"Saved {len(self.static_samples)} samples to {filename}")
            self.static_samples = []
            return filename
        return None
    
    def save_dynamic_data(self):
        """Save collected dynamic gesture data"""
        if len(self.gesture_buffer) > 10:  # Minimum gesture length
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_dir}/dynamic/{self.current_label}_{timestamp}.json"
            
            # Convert deque to list
            gesture_sequence = [frame.tolist() if frame is not None else [0]*63 
                              for frame in self.gesture_buffer]
            
            data = {
                "label": self.current_label,
                "type": "dynamic", 
                "sequence": gesture_sequence,
                "sequence_length": len(gesture_sequence),
                "timestamp": timestamp
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f)
            
            print(f"Saved dynamic gesture sequence to {filename}")
            return filename
        return None
    
    def run_data_collection(self):
        """Interactive data collection interface"""
        cap = cv2.VideoCapture(0)
        
        print("=== GESTURE DATA COLLECTION ===")
        print("Commands:")
        print("  s + letter: Start static recording (e.g., 's a' for letter A)")
        print("  d + word: Start dynamic recording (e.g., 'd hello' for hello gesture)")
        print("  SPACE: Capture static sample")
        print("  r: Start/stop dynamic recording")
        print("  q: Quit")
        
        current_mode = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            landmarks, annotated_frame = self.process_frame(frame)
            
            # Handle dynamic recording
            if self.is_recording and landmarks is not None:
                self.gesture_buffer.append(landmarks)
                cv2.putText(annotated_frame, f"RECORDING {self.current_label}... ({len(self.gesture_buffer)}/60)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Auto-stop after buffer is full
                if len(self.gesture_buffer) >= 60:
                    self.is_recording = False
                    self.save_dynamic_data()
                    print("Recording completed!")
            
            # Show instructions
            if current_mode == "static":
                cv2.putText(annotated_frame, f"Static: {self.current_label} (SPACE to capture)", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"Samples: {len(getattr(self, 'static_samples', []))}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow('Gesture Data Collection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') and current_mode == "static":
                self.capture_static_sample(landmarks)
            elif key == ord('r') and current_mode == "dynamic":
                if not self.is_recording:
                    self.start_dynamic_recording(self.current_label)
                else:
                    self.is_recording = False
                    self.save_dynamic_data()
            elif key == ord('s'):
                # Get static gesture label
                print("Enter static gesture label (e.g., 'A', 'B', 'Hello'): ")
                current_mode = "static"
            elif key == ord('d'): 
                # Get dynamic gesture label
                print("Enter dynamic gesture label (e.g., 'wave', 'thumbs_up'): ")
                current_mode = "dynamic"
            elif key == 13:  # Enter key
                if current_mode == "static" and hasattr(self, 'static_samples'):
                    self.save_static_data()
                    current_mode = None
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    tracker = MediaPipeHandTracker()
    tracker.run_data_collection()

if __name__ == "__main__":
    main()