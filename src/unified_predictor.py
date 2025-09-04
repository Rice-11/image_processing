import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import glob
import os
from src.mediapipe_hand_tracker import MediaPipeHandTracker

class UnifiedGesturePredictor:
    def __init__(self):
        self.hand_tracker = MediaPipeHandTracker()
        
        # Models and encoders
        self.static_model = None
        self.dynamic_model = None
        self.static_classes = None
        self.dynamic_classes = None
        
        # Prediction buffers
        self.landmark_buffer = deque(maxlen=60)  # For dynamic gestures
        self.static_buffer = deque(maxlen=10)    # For static prediction smoothing
        
        # State tracking
        self.is_moving = False
        self.movement_threshold = 0.01
        self.confidence_threshold = 0.7
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load the latest trained models"""
        models_dir = "gesture_models"
        
        # Load static model
        static_models = glob.glob(f"{models_dir}/static_model_*.h5")
        if static_models:
            latest_static = max(static_models, key=os.path.getctime)
            self.static_model = tf.keras.models.load_model(latest_static)
            
            # Load corresponding encoder
            timestamp = latest_static.split('_')[-1].replace('.h5', '')
            encoder_path = f"{models_dir}/static_encoder_{timestamp}.npy"
            if os.path.exists(encoder_path):
                self.static_classes = np.load(encoder_path, allow_pickle=True)
                print(f"Loaded static model: {len(self.static_classes)} classes")
        
        # Load dynamic model
        dynamic_models = glob.glob(f"{models_dir}/dynamic_model_*.h5")
        if dynamic_models:
            latest_dynamic = max(dynamic_models, key=os.path.getctime)
            self.dynamic_model = tf.keras.models.load_model(latest_dynamic)
            
            # Load corresponding encoder
            timestamp = latest_dynamic.split('_')[-1].replace('.h5', '')
            encoder_path = f"{models_dir}/dynamic_encoder_{timestamp}.npy"
            if os.path.exists(encoder_path):
                self.dynamic_classes = np.load(encoder_path, allow_pickle=True)
                print(f"Loaded dynamic model: {len(self.dynamic_classes)} classes")
    
    def detect_movement(self, landmarks):
        """Detect if hand is currently moving"""
        if len(self.landmark_buffer) < 5:
            return False
        
        # Calculate movement by comparing recent frames
        recent_frames = np.array(list(self.landmark_buffer)[-5:])
        movement = np.std(recent_frames, axis=0)
        avg_movement = np.mean(movement)
        
        return avg_movement > self.movement_threshold
    
    def predict_static_gesture(self, landmarks):
        """Predict static gesture from current landmarks"""
        if self.static_model is None or landmarks is None:
            return None, 0.0
        
        # Predict
        prediction = self.static_model.predict(landmarks.reshape(1, -1), verbose=0)
        confidence = np.max(prediction)
        class_idx = np.argmax(prediction)
        
        if confidence >= self.confidence_threshold:
            gesture = self.static_classes[class_idx]
            return gesture, confidence
        
        return None, confidence
    
    def predict_dynamic_gesture(self):
        """Predict dynamic gesture from landmark sequence"""
        if self.dynamic_model is None or len(self.landmark_buffer) < 30:
            return None, 0.0
        
        # Prepare sequence (pad if necessary)
        sequence = list(self.landmark_buffer)
        max_length = self.dynamic_model.input_shape[1]
        
        # Pad with zeros if sequence is shorter than expected
        if len(sequence) < max_length:
            padding = [[0]*63] * (max_length - len(sequence))
            sequence = sequence + padding
        else:
            # Take the most recent max_length frames
            sequence = sequence[-max_length:]
        
        # Convert to numpy array and reshape
        sequence = np.array(sequence).reshape(1, max_length, 63)
        
        # Predict
        prediction = self.dynamic_model.predict(sequence, verbose=0)
        confidence = np.max(prediction)
        class_idx = np.argmax(prediction)
        
        if confidence >= self.confidence_threshold:
            gesture = self.dynamic_classes[class_idx]
            return gesture, confidence
        
        return None, confidence
    
    def run_real_time_prediction(self):
        """Run real-time gesture prediction"""
        if self.static_model is None and self.dynamic_model is None:
            print("❌ No trained models found!")
            print("Please train models first using gesture_trainer.py")
            return
        
        cap = cv2.VideoCapture(0)
        
        print("=== REAL-TIME GESTURE RECOGNITION ===")
        print(f"Static gestures available: {self.static_classes is not None}")
        print(f"Dynamic gestures available: {self.dynamic_classes is not None}")
        print("Press 'q' to quit")
        
        current_prediction = ""
        current_confidence = 0.0
        prediction_type = ""
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmarks, annotated_frame = self.hand_tracker.process_frame(frame)
            
            if landmarks is not None:
                # Add landmarks to buffer
                self.landmark_buffer.append(landmarks)
                
                # Detect movement
                is_currently_moving = self.detect_movement(landmarks)
                
                # Predict based on movement state
                if not is_currently_moving and self.static_model is not None:
                    # Static gesture prediction
                    gesture, confidence = self.predict_static_gesture(landmarks)
                    if gesture:
                        self.static_buffer.append((gesture, confidence))
                        
                        # Smooth static predictions
                        if len(self.static_buffer) >= 5:
                            recent_predictions = [p[0] for p in list(self.static_buffer)[-5:]]
                            most_common = max(set(recent_predictions), key=recent_predictions.count)
                            if recent_predictions.count(most_common) >= 3:
                                current_prediction = most_common
                                current_confidence = confidence
                                prediction_type = "STATIC"
                
                elif is_currently_moving and self.dynamic_model is not None:
                    # Dynamic gesture prediction
                    gesture, confidence = self.predict_dynamic_gesture()
                    if gesture:
                        current_prediction = gesture
                        current_confidence = confidence
                        prediction_type = "DYNAMIC"
                
                # Movement indicator
                movement_color = (0, 0, 255) if is_currently_moving else (0, 255, 0)
                movement_text = "MOVING" if is_currently_moving else "STATIC"
                cv2.putText(annotated_frame, movement_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, movement_color, 2)
            
            # Display current prediction
            if current_prediction:
                # Background for better visibility
                cv2.rectangle(annotated_frame, (10, 60), (400, 120), (0, 0, 0), -1)
                
                # Prediction text
                text = f"{prediction_type}: {current_prediction}"
                cv2.putText(annotated_frame, text, (15, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Confidence
                conf_text = f"Confidence: {current_confidence:.2f}"
                cv2.putText(annotated_frame, conf_text, (15, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Instructions
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Gesture Recognition', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    predictor = UnifiedGesturePredictor()
    predictor.run_real_time_prediction()

if __name__ == "__main__":
    main()