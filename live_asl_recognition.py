#!/usr/bin/env python3
"""
Professional Live ASL Recognition Application
Real-time ASL letter recognition using trained CNN model and MediaPipe hand detection
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import os
import glob
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LiveASLRecognizer:
    def __init__(self, models_dir="trained_models"):
        self.models_dir = models_dir
        
        # Initialize MediaPipe Hand Detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # Model components
        self.model = None
        self.class_mapping = None
        self.class_names = []
        self.metadata = None
        
        # Prediction settings
        self.confidence_threshold = 0.7
        self.prediction_buffer = deque(maxlen=10)  # For prediction smoothing
        
        # Display settings
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.stable_prediction = ""
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.fps_start_time = datetime.now()
        
        # Load the trained model
        self.load_model()
        
        print("Live ASL Recognizer Initialized")
        print(f"Model loaded: {self.model is not None}")
        print(f"Classes available: {len(self.class_names)}")
    
    def find_latest_model(self):
        """Find the most recent trained model"""
        if not os.path.exists(self.models_dir):
            return None, None, None
        
        # First check for 'latest' shortcuts
        latest_model = os.path.join(self.models_dir, 'latest_model.h5')
        latest_mapping = os.path.join(self.models_dir, 'latest_class_mapping.json')
        latest_metadata = os.path.join(self.models_dir, 'latest_metadata.json')
        
        if all(os.path.exists(f) for f in [latest_model, latest_mapping, latest_metadata]):
            return latest_model, latest_mapping, latest_metadata
        
        # Otherwise find the most recent timestamped files
        model_files = glob.glob(os.path.join(self.models_dir, "sign_mnist_model_*.h5"))
        
        if not model_files:
            return None, None, None
        
        # Get the most recent model
        latest_model = max(model_files, key=os.path.getctime)
        
        # Extract timestamp from filename
        timestamp = os.path.basename(latest_model).replace('sign_mnist_model_', '').replace('.h5', '')
        
        # Find corresponding files
        mapping_file = os.path.join(self.models_dir, f"class_mapping_{timestamp}.json")
        metadata_file = os.path.join(self.models_dir, f"training_metadata_{timestamp}.json")
        
        if os.path.exists(mapping_file) and os.path.exists(metadata_file):
            return latest_model, mapping_file, metadata_file
        else:
            return latest_model, None, None
    
    def load_model(self):
        """Load the trained model and associated files"""
        model_path, mapping_path, metadata_path = self.find_latest_model()
        
        if not model_path:
            print("❌ No trained models found!")
            print("Please run 'train_sign_mnist_model.py' first to train a model")
            return False
        
        try:
            print(f"Loading model: {os.path.basename(model_path)}")
            
            # Load the trained model
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully")
            
            # Load class mapping
            if mapping_path and os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.class_mapping = json.load(f)
                self.class_names = self.class_mapping['classes']
                print(f"✅ Class mapping loaded: {len(self.class_names)} classes")
                print(f"Classes: {', '.join(self.class_names)}")
            else:
                # Fallback to default ASL alphabet
                self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                                  'V', 'W', 'X', 'Y']
                print("⚠️ Using default class mapping")
            
            # Load metadata
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Metadata loaded (Test Accuracy: {self.metadata.get('test_accuracy', 'Unknown')})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def extract_hand_region(self, frame, hand_landmarks):
        """Extract and process hand region for model prediction"""
        h, w, c = frame.shape
        
        # Get bounding box coordinates from hand landmarks
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        # Convert normalized coordinates to pixel coordinates
        x_coords = [int(x * w) for x in x_coords]
        y_coords = [int(y * h) for y in y_coords]
        
        # Calculate bounding box with padding
        padding = 40
        x_min = max(0, min(x_coords) - padding)
        x_max = min(w, max(x_coords) + padding)
        y_min = max(0, min(y_coords) - padding)
        y_max = min(h, max(y_coords) + padding)
        
        # Ensure square bounding box for better shape preservation
        box_width = x_max - x_min
        box_height = y_max - y_min
        max_side = max(box_width, box_height)
        
        # Center the square box
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        half_side = max_side // 2
        x_min = max(0, center_x - half_side)
        x_max = min(w, center_x + half_side)
        y_min = max(0, center_y - half_side)
        y_max = min(h, center_y + half_side)
        
        # Extract hand region
        hand_region = frame[y_min:y_max, x_min:x_max]
        
        return hand_region, (x_min, y_min, x_max, y_max)
    
    def preprocess_for_model(self, hand_region):
        """Preprocess hand region to match training data format"""
        if hand_region.size == 0:
            return None
        
        # Convert to grayscale
        if len(hand_region.shape) == 3:
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = hand_region.copy()
        
        # Resize to 28x28 (model input size)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] range (same as training)
        normalized = resized.astype('float32') / 255.0
        
        # Reshape for model input: (1, 28, 28, 1)
        model_input = normalized.reshape(1, 28, 28, 1)
        
        return model_input, resized
    
    def predict_sign(self, processed_image):
        """Predict ASL sign from processed image"""
        if self.model is None or processed_image is None:
            return None, 0.0
        
        try:
            # Get model prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get the class with highest probability
            class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][class_index])
            
            # Get class name
            if class_index < len(self.class_names):
                predicted_class = self.class_names[class_index]
                return predicted_class, confidence
            else:
                return None, confidence
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def smooth_predictions(self, prediction, confidence):
        """Smooth predictions to reduce flickering"""
        # Only add to buffer if confidence is reasonable
        if confidence >= 0.3:  # Lower threshold for buffer
            self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 3:
            return self.stable_prediction
        
        # Get recent high-confidence predictions
        high_conf_predictions = [
            pred for pred, conf in self.prediction_buffer 
            if conf >= self.confidence_threshold
        ]
        
        if not high_conf_predictions:
            return ""
        
        # Find most common prediction
        from collections import Counter
        prediction_counts = Counter(high_conf_predictions)
        
        if prediction_counts:
            most_common = prediction_counts.most_common(1)[0]
            # Only update if prediction appears multiple times
            if most_common[1] >= 2:
                self.stable_prediction = most_common[0]
        
        return self.stable_prediction
    
    def draw_hand_detection(self, frame, hand_landmarks, bbox=None):
        """Draw hand landmarks and bounding box"""
        # Draw hand landmarks
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw_styles.get_default_hand_landmarks_style(),
            self.mp_draw_styles.get_default_hand_connections_style()
        )
        
        # Draw bounding box if provided
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
            cv2.putText(frame, "Hand Region", (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def draw_prediction_ui(self, frame):
        """Draw prediction UI overlay"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for UI
        overlay = frame.copy()
        
        # Main prediction display
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (w - 10, 120), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "ASL Letter Recognition", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Current prediction
        if self.stable_prediction and self.current_confidence >= self.confidence_threshold:
            # Large prediction letter
            cv2.putText(frame, f"Letter: {self.stable_prediction}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Confidence score
            conf_text = f"Confidence: {self.current_confidence:.3f}"
            cv2.putText(frame, conf_text, (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int(300 * self.current_confidence)
            cv2.rectangle(frame, (20, 105), (320, 115), (100, 100, 100), -1)
            if bar_width > 0:
                color = (0, 255, 0) if self.current_confidence >= self.confidence_threshold else (0, 165, 255)
                cv2.rectangle(frame, (20, 105), (20 + bar_width, 115), color, -1)
        
        elif self.current_prediction:
            # Show current prediction even if not stable
            cv2.putText(frame, f"Detecting: {self.current_prediction}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {self.current_confidence:.3f} (Low)", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Place hand in view", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Status information
        status_y = h - 80
        cv2.rectangle(frame, (10, status_y - 5), (w - 10, h - 10), (0, 0, 0), -1)
        
        # Model status
        model_status = "✓ Model Ready" if self.model else "✗ No Model"
        cv2.putText(frame, model_status, (20, status_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.model else (0, 0, 255), 1)
        
        # Performance info
        if hasattr(self, 'fps') and self.fps > 0:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (150, status_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        controls = [
            "Controls:",
            "Q - Quit",
            "C - Adjust confidence",
            "R - Reset predictions",
            "S - Save screenshot"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (w - 200, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_processed_hand_preview(self, frame, processed_hand, bbox):
        """Draw small preview of processed hand region"""
        if processed_hand is not None and bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            
            # Scale up the 28x28 image for visibility
            preview_size = 100
            preview = cv2.resize(processed_hand, (preview_size, preview_size), 
                               interpolation=cv2.INTER_NEAREST)
            
            # Convert to BGR for display
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            
            # Position preview in top-right corner
            h, w = frame.shape[:2]
            preview_x = w - preview_size - 20
            preview_y = 130
            
            # Add preview to frame
            frame[preview_y:preview_y + preview_size, 
                  preview_x:preview_x + preview_size] = preview_bgr
            
            # Add border and label
            cv2.rectangle(frame, (preview_x, preview_y), 
                         (preview_x + preview_size, preview_y + preview_size), 
                         (255, 255, 255), 2)
            cv2.putText(frame, "Model Input", (preview_x, preview_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:  # Update every 30 frames
            current_time = datetime.now()
            time_diff = (current_time - self.fps_start_time).total_seconds()
            self.fps = 30 / time_diff
            self.fps_start_time = current_time
    
    def run_live_recognition(self):
        """Main live recognition loop"""
        if not self.model:
            print("❌ No model loaded. Please train a model first.")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*70)
        print("LIVE ASL RECOGNITION STARTED")
        print("="*70)
        print("Instructions:")
        print("• Make clear ASL letter signs with your hand")
        print("• Keep hand centered and well-lit")
        print("• Wait for stable predictions (green text)")
        print("• Available letters: A-Y (excluding J, Z)")
        print("• Press 'Q' to quit, 'C' to adjust confidence")
        print("="*70)
        
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Could not read frame")
                    break
                
                # Flip frame for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Process hand detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Reset prediction if no hand detected
                hand_detected = False
                processed_hand_preview = None
                bbox = None
                
                if results.multi_hand_landmarks:
                    hand_detected = True
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand detection
                        hand_region, bbox = self.extract_hand_region(frame, hand_landmarks)
                        self.draw_hand_detection(frame, hand_landmarks, bbox)
                        
                        # Process hand region for model
                        model_input, processed_hand_preview = self.preprocess_for_model(hand_region)
                        
                        if model_input is not None:
                            # Get prediction
                            prediction, confidence = self.predict_sign(model_input)
                            
                            if prediction:
                                self.current_prediction = prediction
                                self.current_confidence = confidence
                                
                                # Update stable prediction
                                self.stable_prediction = self.smooth_predictions(prediction, confidence)
                
                # Clear old predictions if no hand detected
                if not hand_detected:
                    self.current_prediction = ""
                    self.current_confidence = 0.0
                    # Don't clear stable prediction immediately - let it fade naturally
                
                # Draw UI elements
                self.draw_prediction_ui(frame)
                
                # Draw processed hand preview
                if processed_hand_preview is not None:
                    self.draw_processed_hand_preview(frame, processed_hand_preview, bbox)
                
                # Calculate and display FPS
                self.calculate_fps()
                
                # Display frame
                cv2.imshow('Live ASL Recognition', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Adjust confidence threshold
                    print(f"\nCurrent confidence threshold: {self.confidence_threshold}")
                    try:
                        new_threshold = float(input("Enter new threshold (0.1-1.0): "))
                        if 0.1 <= new_threshold <= 1.0:
                            self.confidence_threshold = new_threshold
                            print(f"Confidence threshold set to: {self.confidence_threshold}")
                        else:
                            print("Invalid threshold. Must be between 0.1 and 1.0")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                elif key == ord('r'):
                    # Reset predictions
                    self.prediction_buffer.clear()
                    self.stable_prediction = ""
                    self.current_prediction = ""
                    print("Predictions reset")
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_count += 1
                    screenshot_path = f"screenshot_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\nStopping recognition...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n🎯 Live ASL Recognition ended")
            print(f"Total frames processed: {self.frame_count}")
            if hasattr(self, 'fps') and self.fps > 0:
                print(f"Average FPS: {self.fps:.1f}")

def main():
    """Main application entry point"""
    print("Live ASL Recognition Application")
    print("Real-time ASL letter recognition using CNN and MediaPipe")
    print()
    
    # Check for trained models
    models_dir = "trained_models"
    if not os.path.exists(models_dir):
        print("❌ No models directory found!")
        print("Please run 'python train_sign_mnist_model.py' first to train a model")
        return
    
    # Check for model files
    model_files = glob.glob(os.path.join(models_dir, "*.h5"))
    if not model_files:
        print("❌ No trained models found!")
        print("Please run 'python train_sign_mnist_model.py' first to train a model")
        return
    
    # Initialize recognizer
    recognizer = LiveASLRecognizer(models_dir)
    
    if recognizer.model:
        print("✅ System ready for live recognition!")
        print("\nModel Information:")
        if recognizer.metadata:
            print(f"  Test Accuracy: {recognizer.metadata.get('test_accuracy', 'Unknown')}")
            print(f"  Training Date: {recognizer.metadata.get('timestamp', 'Unknown')}")
        print(f"  Available Classes: {len(recognizer.class_names)}")
        print()
        
        input("Press Enter to start the camera...")
        recognizer.run_live_recognition()
    else:
        print("❌ Failed to load model. Please check your trained models.")

if __name__ == "__main__":
    main()