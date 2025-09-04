import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os
from datetime import datetime
import mediapipe as mp

from src.data_loader import SignLanguageDataLoader

class SignLanguagePredictor:
    def __init__(self, model_path=None, confidence_threshold=0.7):
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.data_loader = SignLanguageDataLoader()
        self.class_names = self.data_loader.class_names
        
        # --- Initialize MediaPipe Hand Detector ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        self.hand_detector_loaded = True
        print("MediaPipe hand detector initialized successfully")

        # For smooth predictions
        self.prediction_buffer = deque(maxlen=15) # Increased buffer size for more stability
        self.last_stable_prediction = ""
        self.last_confidence = 0.0
        
        # Camera settings
        self.frame_count = 0
        self.fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Screenshot settings
        self.screenshot_dir = "screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No sign classification model path provided or model not found.")
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Sign classification model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading sign classification model: {e}")
            self.model = None
    
    def list_available_models(self):
        """List all available trained models"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            print("No models directory found.")
            return
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        if not model_files:
            print("No trained models found in models directory.")
            return
        
        print("Available models:")
        for i, model_file in enumerate(model_files, 1):
            print(f"  {i}. {model_file}")
        
        return model_files
    
    def run_real_time_prediction(self, camera_index=0):
        """Run real-time sign language prediction with hand detection."""
        
        # --- Final check for required models ---
        if not self.hand_detector_loaded:
            print("\nCRITICAL ERROR: Cannot start prediction.")
            print("The Hand Detection model is missing or failed to load.")
            print("Please scroll up and follow the instructions to download the required files.")
            return
            
        if self.model is None:
            print("\nERROR: Cannot start prediction. The Sign Classification model is not loaded.")
            print("Please choose a valid model when prompted.")
            return

        print("Starting real-time sign language recognition...")
        print("Controls: Q - Quit")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # --- MediaPipe Hand Detection and Sign Prediction ---
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Create annotated frame
                annotated_frame = frame.copy()
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw_styles.get_default_hand_landmarks_style(),
                            self.mp_draw_styles.get_default_hand_connections_style()
                        )
                        
                        # Get bounding box from landmarks
                        h, w, c = frame.shape
                        x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                        x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                        y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                        y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
                        
                        # Add margin for better extraction
                        margin = 20
                        x1 = max(0, x_min - margin)
                        y1 = max(0, y_min - margin)
                        x2 = min(w, x_max + margin)
                        y2 = min(h, y_max + margin)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, "HAND", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Extract hand ROI for prediction
                        hand_roi = frame[y1:y2, x1:x2]
                        
                        if hand_roi.size > 0 and self.model is not None:
                            # Preprocess the ROI for the sign language model
                            preprocessed_roi = self.data_loader.preprocess_camera_frame(hand_roi)
                            
                            # Predict the sign
                            prediction = self.model.predict(preprocessed_roi)
                            confidence = np.max(prediction)
                            
                            if confidence >= self.confidence_threshold:
                                predicted_class_idx = np.argmax(prediction)
                                predicted_class_name = self.class_names[predicted_class_idx]
                                self.prediction_buffer.append(predicted_class_name)
                            else:
                                self.prediction_buffer.append(None)
                        else:
                            self.prediction_buffer.append(None)
                else:
                    # No hands detected
                    self.prediction_buffer.append(None)
                    cv2.putText(annotated_frame, "No hands detected", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # --- Smoothing and Display ---
                if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
                    # Get the most common prediction in the buffer
                    valid_preds = [p for p in self.prediction_buffer if p is not None]
                    if valid_preds:
                        most_common_pred = max(set(valid_preds), key=valid_preds.count)
                        # Only update if the prediction is frequent enough
                        if valid_preds.count(most_common_pred) > 5: # Stability threshold
                            self.last_stable_prediction = most_common_pred
                    else:
                        self.last_stable_prediction = ""

                # Display the stable prediction
                if self.last_stable_prediction:
                    cv2.putText(annotated_frame, f"Prediction: {self.last_stable_prediction}", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)

                cv2.imshow('Sign Language Recognition', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\nStopping prediction...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

def main():
    """Main function for running camera prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Sign Language Recognition')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--confidence', type=float, default=0.7, 
                       help='Confidence threshold for predictions')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--image', type=str, help='Test single image instead of camera')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = SignLanguagePredictor(args.model_path, args.confidence)
    
    if args.image:
        # Test single image
        # predictor.test_single_prediction(args.image)
        print("Image testing not implemented in this version.")
    else:
        # Run real-time prediction
        predictor.run_real_time_prediction(args.camera)

if __name__ == "__main__":
    main()
