import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os
from datetime import datetime

from data_loader import SignLanguageDataLoader

class SignLanguagePredictor:
    def __init__(self, model_path=None, confidence_threshold=0.7):
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.data_loader = SignLanguageDataLoader()
        self.class_names = self.data_loader.class_names
        
        # For smooth predictions
        self.prediction_buffer = deque(maxlen=10)
        self.last_prediction = ""
        self.last_confidence = 0.0
        
        # Camera settings
        self.roi_size = 200
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
            print("No model path provided or model not found.")
            print("Available models:")
            self.list_available_models()
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from: {model_path}")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
        except Exception as e:
            print(f"Error loading model: {e}")
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
    
    def preprocess_frame(self, frame):
        """Preprocess camera frame for prediction"""
        # Extract ROI
        roi, roi_coords = self.data_loader.extract_hand_roi(frame, self.roi_size)
        
        # Preprocess for model
        processed = self.data_loader.preprocess_camera_frame(roi)
        
        return processed, roi, roi_coords
    
    def predict_sign(self, frame):
        """Predict sign language letter from frame"""
        if self.model is None:
            return "No Model", 0.0, frame
        
        # Preprocess frame
        processed_frame, roi, roi_coords = self.preprocess_frame(frame)
        
        # Make prediction
        try:
            predictions = self.model.predict(processed_frame, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get predicted letter
            predicted_letter = self.class_names[predicted_class]
            
            # Add to buffer for smoothing
            if confidence > self.confidence_threshold:
                self.prediction_buffer.append(predicted_letter)
            
            # Get smoothed prediction
            smoothed_prediction = self.get_smoothed_prediction()
            
            return smoothed_prediction, confidence, roi
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0, frame
    
    def get_smoothed_prediction(self):
        """Get smoothed prediction from buffer"""
        if not self.prediction_buffer:
            return "No Prediction"
        
        # Find most common prediction in buffer
        unique, counts = np.unique(list(self.prediction_buffer), return_counts=True)
        most_common_idx = np.argmax(counts)
        
        # Only return if it appears frequently enough
        if counts[most_common_idx] >= len(self.prediction_buffer) * 0.6:
            return unique[most_common_idx]
        else:
            return self.last_prediction if self.last_prediction else "Uncertain"
    
    def draw_ui(self, frame, prediction, confidence, roi_coords):
        """Draw user interface on frame"""
        height, width = frame.shape[:2]
        
        # Draw ROI rectangle
        x1, y1, x2, y2 = roi_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand here", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw prediction
        prediction_text = f"Prediction: {prediction}"
        confidence_text = f"Confidence: {confidence:.2f}"
        
        # Background for text
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Prediction text
        cv2.putText(frame, prediction_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Confidence text with color coding
        conf_color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 165, 255)
        cv2.putText(frame, confidence_text, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Controls:",
            "Q - Quit",
            "R - Reset",
            "S - Screenshot",
            "C - Toggle confidence"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (width - 200, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def save_screenshot(self, frame, prediction, confidence):
        """Save screenshot with prediction"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{prediction}_{confidence:.2f}_{timestamp}.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)
        
        cv2.imwrite(filepath, frame)
        print(f"Screenshot saved: {filepath}")
    
    def run_real_time_prediction(self, camera_index=0):
        """Run real-time sign language prediction"""
        if self.model is None:
            print("No model loaded. Cannot start real-time prediction.")
            return
        
        print("Starting real-time sign language recognition...")
        print("Controls:")
        print("  Q - Quit")
        print("  R - Reset prediction buffer")
        print("  S - Save screenshot")
        print("  C - Toggle confidence threshold")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera initialized successfully")
        print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Make prediction
                prediction, confidence, roi = self.predict_sign(frame)
                
                # Update last prediction if confident
                if confidence > self.confidence_threshold:
                    self.last_prediction = prediction
                    self.last_confidence = confidence
                
                # Draw UI
                frame = self.draw_ui(frame, prediction, confidence, 
                                   self.get_roi_coords(frame))
                
                # Show ROI in corner
                if roi is not None and roi.size > 0:
                    roi_resized = cv2.resize(roi, (150, 150))
                    frame[10:160, frame.shape[1]-160:frame.shape[1]-10] = roi_resized
                    cv2.rectangle(frame, (frame.shape[1]-160, 10), (frame.shape[1]-10, 160), (255, 255, 255), 2)
                    cv2.putText(frame, "ROI", (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Display frame
                cv2.imshow('Sign Language Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('r'):  # Reset buffer
                    self.prediction_buffer.clear()
                    self.last_prediction = ""
                    print("Prediction buffer reset")
                elif key == ord('s'):  # Save screenshot
                    self.save_screenshot(frame, prediction, confidence)
                elif key == ord('c'):  # Toggle confidence threshold
                    self.confidence_threshold = 0.8 if self.confidence_threshold == 0.7 else 0.7
                    print(f"Confidence threshold: {self.confidence_threshold}")
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\nStopping prediction...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")
    
    def get_roi_coords(self, frame):
        """Get ROI coordinates for current frame"""
        height, width = frame.shape[:2]
        x1 = (width - self.roi_size) // 2
        y1 = (height - self.roi_size) // 2
        x2 = x1 + self.roi_size
        y2 = y1 + self.roi_size
        return (x1, y1, x2, y2)
    
    def test_single_prediction(self, image_path):
        """Test prediction on a single image"""
        if self.model is None:
            print("No model loaded.")
            return
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Make prediction
        prediction, confidence, roi = self.predict_sign(img)
        
        print(f"Image: {image_path}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        
        # Display result
        img = self.draw_ui(img, prediction, confidence, self.get_roi_coords(img))
        cv2.imshow('Single Image Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        predictor.test_single_prediction(args.image)
    else:
        # Run real-time prediction
        predictor.run_real_time_prediction(args.camera)

if __name__ == "__main__":
    main()