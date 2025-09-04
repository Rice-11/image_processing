#!/usr/bin/env python3
"""
Real-Time ASL Prediction Application
Live ASL letter recognition using trained model and MediaPipe hand tracking
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
import glob
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ASLLivePredictor:
    def __init__(self, models_dir="asl_models"):
        self.models_dir = models_dir
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # Prediction settings
        self.model = None
        self.label_encoder_data = None
        self.normalization_params = None
        self.confidence_threshold = 0.7
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=15)  # Buffer for smoothing predictions
        self.confidence_buffer = deque(maxlen=15)
        
        # Display settings
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.stable_prediction = ""
        
        # Load the latest trained model
        self.load_latest_model()
        
        print("ASL Live Predictor Initialized")
        print(f"Model loaded: {self.model is not None}")
        print(f"Confidence threshold: {self.confidence_threshold}")
    
    def find_latest_model(self):
        """Find the most recent trained model"""
        if not os.path.exists(self.models_dir):
            return None, None, None
        
        # Find all model files
        model_files = glob.glob(os.path.join(self.models_dir, "asl_model_*.h5"))
        
        if not model_files:
            return None, None, None
        
        # Get the most recent model
        latest_model = max(model_files, key=os.path.getctime)
        
        # Extract timestamp from filename
        timestamp = os.path.basename(latest_model).replace('asl_model_', '').replace('.h5', '')
        
        # Find corresponding encoder and normalization files
        encoder_file = os.path.join(self.models_dir, f"label_encoder_{timestamp}.json")
        norm_file = os.path.join(self.models_dir, f"normalization_{timestamp}.json")
        
        if os.path.exists(encoder_file) and os.path.exists(norm_file):
            return latest_model, encoder_file, norm_file
        else:
            return latest_model, None, None
    
    def load_latest_model(self):
        """Load the most recent trained model and its parameters"""
        model_path, encoder_path, norm_path = self.find_latest_model()
        
        if not model_path:
            print("❌ No trained models found!")
            print("Please run 'asl_model_trainer.py' first to train a model")
            return False
        
        try:
            # Load model
            print(f"Loading model: {os.path.basename(model_path)}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Load label encoder
            if encoder_path:
                with open(encoder_path, 'r') as f:
                    self.label_encoder_data = json.load(f)
                print(f"Loaded classes: {self.label_encoder_data['classes']}")
            
            # Load normalization parameters
            if norm_path:
                with open(norm_path, 'r') as f:
                    norm_data = json.load(f)
                    self.normalization_params = {
                        'mean': np.array(norm_data['mean']),
                        'std': np.array(norm_data['std'])
                    }
                print("Normalization parameters loaded")
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def extract_hand_landmarks(self, hand_landmarks):
        """Extract normalized hand landmarks as feature vector"""
        landmarks = []
        
        # Get wrist position as reference point
        wrist = hand_landmarks.landmark[0]
        
        # Extract all 21 landmarks relative to wrist
        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x - wrist.x,  # Relative x position
                landmark.y - wrist.y,  # Relative y position  
                landmark.z - wrist.z   # Relative z position
            ])
        
        return np.array(landmarks, dtype=np.float32)
    
    def normalize_features(self, landmarks):
        """Normalize landmarks using training parameters"""
        if self.normalization_params is None:
            return landmarks
        
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        
        normalized = (landmarks - mean) / std
        return normalized
    
    def predict_sign(self, landmarks):
        """Predict ASL sign from landmarks"""
        if self.model is None or self.label_encoder_data is None:
            return None, 0.0
        
        try:
            # Normalize landmarks
            normalized_landmarks = self.normalize_features(landmarks)
            
            # Reshape for model input
            input_data = normalized_landmarks.reshape(1, -1)
            
            # Get prediction
            prediction = self.model.predict(input_data, verbose=0)
            
            # Get class with highest probability
            class_index = np.argmax(prediction)
            confidence = float(prediction[0][class_index])
            
            # Get class name
            class_name = self.label_encoder_data['classes'][class_index]
            
            return class_name, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def smooth_predictions(self, prediction, confidence):
        """Smooth predictions using a buffer to reduce flickering"""
        # Add to buffers
        self.prediction_buffer.append(prediction)
        self.confidence_buffer.append(confidence)
        
        # Only proceed if we have enough predictions
        if len(self.prediction_buffer) < 5:
            return self.stable_prediction
        
        # Filter by confidence threshold
        valid_predictions = []
        for i, conf in enumerate(self.confidence_buffer):
            if conf >= self.confidence_threshold:
                valid_predictions.append(self.prediction_buffer[i])
        
        if not valid_predictions:
            return ""
        
        # Find most common prediction
        from collections import Counter
        prediction_counts = Counter(valid_predictions)
        most_common = prediction_counts.most_common(1)[0]
        
        # Only update if prediction appears frequently enough
        if most_common[1] >= 3:  # Appears at least 3 times in recent predictions
            self.stable_prediction = most_common[0]
        
        return self.stable_prediction
    
    def draw_prediction_ui(self, frame):
        """Draw prediction UI on frame"""
        h, w = frame.shape[:2]
        
        # Background for prediction display
        cv2.rectangle(frame, (10, 10), (w-10, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w-10, 150), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "ASL Live Recognition", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Current prediction
        if self.stable_prediction:
            # Large prediction letter
            cv2.putText(frame, f"Sign: {self.stable_prediction}", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Confidence
            conf_text = f"Confidence: {self.current_confidence:.2f}"
            cv2.putText(frame, conf_text, (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int(300 * self.current_confidence)
            cv2.rectangle(frame, (20, 125), (320, 140), (100, 100, 100), -1)
            if bar_width > 0:
                color = (0, 255, 0) if self.current_confidence >= self.confidence_threshold else (0, 165, 255)
                cv2.rectangle(frame, (20, 125), (20 + bar_width, 140), color, -1)
        else:
            cv2.putText(frame, "No sign detected", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Status info
        status_y = h - 60
        if self.model:
            cv2.putText(frame, "✓ Model loaded", (20, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "✗ No model loaded", (20, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Instructions
        instructions = [
            "Instructions:",
            "• Make clear ASL letters",
            "• Keep hand steady",
            "• Press 'c' to adjust confidence",
            "• Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (w - 280, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Draw enhanced hand landmarks"""
        # Draw standard landmarks
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw_styles.get_default_hand_landmarks_style(),
            self.mp_draw_styles.get_default_hand_connections_style()
        )
        
        # Draw bounding box
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Highlight key landmarks (wrist, fingertips)
        key_points = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
        for point_id in key_points:
            landmark = hand_landmarks.landmark[point_id]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
    
    def run_live_prediction(self):
        """Main live prediction loop"""
        if not self.model:
            print("❌ No model loaded. Please train a model first.")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("ASL LIVE PREDICTION STARTED")
        print("="*60)
        print("Instructions:")
        print("• Make clear ASL letter signs with your hand")
        print("• Keep your hand steady for best results")
        print("• Prediction will appear when confident enough")
        print("• Press 'c' to change confidence threshold")
        print("• Press 'q' to quit")
        print("="*60)
        
        frame_count = 0
        fps_counter = 0
        fps_start_time = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read frame")
                break
            
            frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            
            # Process hand detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Reset prediction if no hand detected
            hand_detected = False
            
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw enhanced landmarks
                    self.draw_hand_landmarks(frame, hand_landmarks)
                    
                    # Extract landmarks and predict
                    landmarks = self.extract_hand_landmarks(hand_landmarks)
                    prediction, confidence = self.predict_sign(landmarks)
                    
                    if prediction and confidence >= 0.5:  # Lower threshold for display
                        self.current_prediction = prediction
                        self.current_confidence = confidence
                        
                        # Smooth predictions
                        self.stable_prediction = self.smooth_predictions(prediction, confidence)
            
            # Clear prediction if no hand for too long
            if not hand_detected:
                self.current_prediction = ""
                self.current_confidence = 0.0
                if frame_count % 30 == 0:  # Every second at 30fps
                    self.stable_prediction = ""
            
            # Draw UI
            self.draw_prediction_ui(frame)
            
            # FPS calculation
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = datetime.now()
                fps = 30 / (fps_end_time - fps_start_time).total_seconds()
                fps_start_time = fps_end_time
                fps_counter = 0
                
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('ASL Live Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Adjust confidence threshold
                print(f"Current confidence threshold: {self.confidence_threshold}")
                try:
                    new_threshold = float(input("Enter new threshold (0.1-1.0): "))
                    if 0.1 <= new_threshold <= 1.0:
                        self.confidence_threshold = new_threshold
                        print(f"Confidence threshold set to: {self.confidence_threshold}")
                    else:
                        print("Invalid threshold. Must be between 0.1 and 1.0")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n🎯 ASL Live Recognition ended")
        print(f"Total frames processed: {frame_count}")

def main():
    """Main application entry point"""
    print("ASL Live Recognition Application")
    print("Real-time ASL letter recognition using MediaPipe and Neural Networks")
    print()
    
    # Check for trained models
    models_dir = "asl_models"
    if not os.path.exists(models_dir) or not glob.glob(os.path.join(models_dir, "*.h5")):
        print("❌ No trained models found!")
        print("Please follow these steps:")
        print("1. Run 'python asl_data_collector.py' to collect training data")
        print("2. Run 'python asl_model_trainer.py' to train the model")
        print("3. Then run this script for live recognition")
        return
    
    # Initialize and run predictor
    predictor = ASLLivePredictor()
    
    if predictor.model:
        print("✅ Ready for live recognition!")
        input("Press Enter to start the camera...")
        predictor.run_live_prediction()
    else:
        print("❌ Failed to load model. Please check your trained models.")

if __name__ == "__main__":
    main()