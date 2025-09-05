#!/usr/bin/env python3
"""
Final ASL Recognition Script with Inversion Fix
Corrected version that properly handles the black hand on white background vs 
white hand on black background inversion issue
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

class FinalASLRecognizer:
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
        self.prediction_buffer = deque(maxlen=12)
        
        # Display settings
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.stable_prediction = ""
        
        # CORRECTED Image processing parameters
        self.blur_kernel_size = 5
        self.adaptive_thresh_block_size = 11
        self.adaptive_thresh_c = 2
        self.hand_padding = 50
        
        # KEY FIX: Use THRESH_BINARY_INV to invert the thresholding
        # This makes the hand WHITE on BLACK background (like MNIST)
        self.use_inverted_threshold = True
        
        # Debug visualization
        self.processed_image_for_debug = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.fps_start_time = datetime.now()
        
        # Load the trained model
        self.load_model()
        
        print("Final ASL Recognizer Initialized with INVERSION FIX")
        print(f"Model loaded: {self.model is not None}")
        print(f"Classes available: {len(self.class_names)}")
        print("🔧 CORRECTED: Using THRESH_BINARY_INV for proper hand inversion")
    
    def find_latest_model(self):
        """Find the most recent trained model"""
        if not os.path.exists(self.models_dir):
            return None, None, None
        
        # Check for 'latest' shortcuts first
        latest_model = os.path.join(self.models_dir, 'latest_model.h5')
        latest_mapping = os.path.join(self.models_dir, 'latest_class_mapping.json')
        latest_metadata = os.path.join(self.models_dir, 'latest_metadata.json')
        
        if all(os.path.exists(f) for f in [latest_model, latest_mapping, latest_metadata]):
            return latest_model, latest_mapping, latest_metadata
        
        # Otherwise find most recent timestamped files
        model_files = glob.glob(os.path.join(self.models_dir, "sign_mnist_model_*.h5"))
        if not model_files:
            return None, None, None
        
        latest_model = max(model_files, key=os.path.getctime)
        timestamp = os.path.basename(latest_model).replace('sign_mnist_model_', '').replace('.h5', '')
        
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
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully")
            
            # Load class mapping
            if mapping_path and os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.class_mapping = json.load(f)
                self.class_names = self.class_mapping['classes']
                print(f"✅ Class mapping loaded: {len(self.class_names)} classes")
            else:
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
    
    def extract_hand_region_robust(self, frame, hand_landmarks):
        """Extract hand region with robust bounding box calculation"""
        h, w, c = frame.shape
        
        # Get landmark coordinates
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        # Convert to pixel coordinates
        x_pixels = [int(x * w) for x in x_coords]
        y_pixels = [int(y * h) for y in y_coords]
        
        # Calculate bounding box with generous padding
        padding = self.hand_padding
        x_min = max(0, min(x_pixels) - padding)
        x_max = min(w, max(x_pixels) + padding)
        y_min = max(0, min(y_pixels) - padding)
        y_max = min(h, max(y_pixels) + padding)
        
        # Ensure square aspect ratio for better shape preservation
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
    
    def corrected_image_preprocessing(self, hand_region):
        """
        CORRECTED preprocessing with proper inversion to match MNIST format:
        This creates WHITE hand on BLACK background (same as MNIST training data)
        """
        if hand_region.size == 0:
            return None, None
        
        # Step 1: Convert to grayscale
        if len(hand_region.shape) == 3:
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = hand_region.copy()
        
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Step 3: CORRECTED adaptive thresholding
        # KEY FIX: Using THRESH_BINARY_INV instead of THRESH_BINARY
        # This inverts the result so hand becomes WHITE on BLACK background (like MNIST)
        if self.use_inverted_threshold:
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred,
                255,  # Max value
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method
                cv2.THRESH_BINARY_INV,  # INVERTED thresholding - KEY FIX!
                self.adaptive_thresh_block_size,  # Block size
                self.adaptive_thresh_c  # Constant
            )
        else:
            # Fallback to original method
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.adaptive_thresh_block_size,
                self.adaptive_thresh_c
            )
        
        # Step 4: Additional morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        
        # Remove small noise
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Step 5: Resize to 28x28 (MNIST size)
        resized = cv2.resize(cleaned, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Step 6: Normalize to [0, 1] range (same as training)
        normalized = resized.astype('float32') / 255.0
        
        # Step 7: Reshape for model input (1, 28, 28, 1)
        model_input = normalized.reshape(1, 28, 28, 1)
        
        # Store processed image for debug visualization
        self.processed_image_for_debug = resized
        
        return model_input, resized
    
    def predict_sign_enhanced(self, processed_image):
        """Enhanced prediction with confidence analysis"""
        if self.model is None or processed_image is None:
            return None, 0.0, []
        
        try:
            # Get model prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get prediction probabilities
            class_probabilities = predictions[0]
            
            # Get the class with highest probability
            class_index = np.argmax(class_probabilities)
            confidence = float(class_probabilities[class_index])
            
            # Get top 3 predictions for debugging
            top_3_indices = np.argsort(class_probabilities)[-3:][::-1]
            top_3_predictions = []
            for idx in top_3_indices:
                if idx < len(self.class_names):
                    top_3_predictions.append({
                        'letter': self.class_names[idx],
                        'confidence': float(class_probabilities[idx])
                    })
            
            # Get predicted class name
            if class_index < len(self.class_names):
                predicted_class = self.class_names[class_index]
                return predicted_class, confidence, top_3_predictions
            else:
                return None, confidence, top_3_predictions
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, []
    
    def smooth_predictions_enhanced(self, prediction, confidence):
        """Enhanced prediction smoothing with better stability"""
        # Only add to buffer if confidence is reasonable
        if confidence >= 0.3:  # Lower threshold for buffer entry
            self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 6:
            return self.stable_prediction
        
        # Get recent high-confidence predictions
        high_conf_predictions = [
            pred for pred, conf in self.prediction_buffer 
            if conf >= self.confidence_threshold
        ]
        
        if not high_conf_predictions:
            # If no high confidence predictions, check for consistent medium-confidence ones
            medium_conf_predictions = [
                pred for pred, conf in list(self.prediction_buffer)[-6:] 
                if conf >= 0.5
            ]
            
            if medium_conf_predictions:
                from collections import Counter
                prediction_counts = Counter(medium_conf_predictions)
                most_common = prediction_counts.most_common(1)[0]
                if most_common[1] >= 4:  # Appears at least 4 times in last 6
                    return most_common[0]
            
            return ""
        
        # Find most common high-confidence prediction
        from collections import Counter
        prediction_counts = Counter(high_conf_predictions)
        
        if prediction_counts:
            most_common = prediction_counts.most_common(1)[0]
            if most_common[1] >= 3:  # Appears at least 3 times
                self.stable_prediction = most_common[0]
        
        return self.stable_prediction
    
    def draw_enhanced_hand_detection(self, frame, hand_landmarks, bbox=None):
        """Enhanced hand detection visualization"""
        # Draw hand landmarks with better visibility
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=3, circle_radius=4
            ),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2
            )
        )
        
        # Draw enhanced bounding box
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            # Main bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
            
            # Corner markers
            corner_size = 25
            for (x, y) in [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]:
                cv2.line(frame, (x-corner_size//2, y), (x+corner_size//2, y), (0, 255, 255), 4)
                cv2.line(frame, (x, y-corner_size//2), (x, y+corner_size//2), (0, 255, 255), 4)
            
            # Label
            cv2.putText(frame, "Hand Detection", (x_min, y_min - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    def draw_final_prediction_ui(self, frame, top_3_predictions=None):
        """Final enhanced prediction UI"""
        h, w = frame.shape[:2]
        
        # Main prediction panel
        panel_height = 180
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (255, 255, 255), 3)
        
        # Title with fix indicator
        cv2.putText(frame, "FINAL ASL Recognition (INVERSION FIXED)", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Main prediction with enhanced styling
        if self.stable_prediction and self.current_confidence >= self.confidence_threshold:
            # Stable prediction (bright green)
            cv2.putText(frame, f"Letter: {self.stable_prediction}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
            
            # Confidence with color coding
            conf_color = (0, 255, 0) if self.current_confidence >= 0.9 else (0, 200, 200)
            cv2.putText(frame, f"Confidence: {self.current_confidence:.3f}", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
            
            # Confidence bar
            bar_width = int(400 * self.current_confidence)
            cv2.rectangle(frame, (20, 120), (420, 135), (100, 100, 100), -1)
            if bar_width > 0:
                bar_color = (0, 255, 0) if self.current_confidence >= 0.8 else (0, 255, 255)
                cv2.rectangle(frame, (20, 120), (20 + bar_width, 135), bar_color, -1)
            
        elif self.current_prediction:
            # Current prediction (yellow)
            cv2.putText(frame, f"Detecting: {self.current_prediction}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, f"Confidence: {self.current_confidence:.3f} (Building...)", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Place hand in clear view", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        
        # Top 3 predictions for debugging
        if top_3_predictions and len(top_3_predictions) >= 3:
            cv2.putText(frame, "Top 3:", (20, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            for i, pred in enumerate(top_3_predictions[:3]):
                pred_text = f"{pred['letter']}:{pred['confidence']:.2f}"
                cv2.putText(frame, pred_text, (80 + i * 80, 155), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Processing status
        status_y = h - 60
        cv2.rectangle(frame, (10, status_y - 5), (w - 10, h - 10), (0, 0, 0), -1)
        
        # Fix indicator
        fix_text = "✅ INVERSION CORRECTED: White hand on black background"
        cv2.putText(frame, fix_text, (20, status_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Performance info
        if hasattr(self, 'fps') and self.fps > 0:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 100, status_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        controls = [
            "Controls:",
            "Q - Quit", "C - Confidence", "T - Toggle inversion",
            "R - Reset", "S - Screenshot", "D - Debug window"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (w - 220, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    def create_enhanced_debug_window(self):
        """Enhanced debug visualization window"""
        if self.processed_image_for_debug is not None:
            # Scale up the 28x28 image
            debug_size = 400
            debug_image = cv2.resize(
                self.processed_image_for_debug, 
                (debug_size, debug_size), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Convert to BGR for display
            debug_bgr = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
            
            # Add grid lines
            grid_step = debug_size // 28
            for i in range(0, debug_size, grid_step):
                cv2.line(debug_bgr, (i, 0), (i, debug_size), (0, 100, 0), 1)
                cv2.line(debug_bgr, (0, i), (debug_size, i), (0, 100, 0), 1)
            
            # Add comprehensive information panel
            info_panel = np.zeros((120, debug_size, 3), dtype=np.uint8)
            
            # Title
            cv2.putText(info_panel, "CORRECTED Model Input (28x28)", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Statistics
            cv2.putText(info_panel, f"Min: {self.processed_image_for_debug.min()}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, f"Max: {self.processed_image_for_debug.max()}", (150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Format explanation
            cv2.putText(info_panel, "WHITE = Hand, BLACK = Background", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(info_panel, "(Same format as MNIST training data)", (10, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Inversion status
            inversion_status = "THRESH_BINARY_INV" if self.use_inverted_threshold else "THRESH_BINARY"
            cv2.putText(info_panel, f"Threshold: {inversion_status}", (10, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Combine panels
            combined_debug = np.vstack([info_panel, debug_bgr])
            
            cv2.imshow('CORRECTED Model Input Debug', combined_debug)
        else:
            # Show placeholder
            placeholder = np.zeros((520, 400, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No hand detected", (100, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('CORRECTED Model Input Debug', placeholder)
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = datetime.now()
            time_diff = (current_time - self.fps_start_time).total_seconds()
            self.fps = 30 / time_diff
            self.fps_start_time = current_time
    
    def run_final_recognition(self):
        """Main recognition loop with all fixes applied"""
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
        
        print("\n" + "="*80)
        print("🎯 FINAL ASL RECOGNITION - INVERSION ISSUE FIXED!")
        print("="*80)
        print("Key Fixes Applied:")
        print("✅ Using cv2.THRESH_BINARY_INV for proper hand inversion")
        print("✅ White hand on black background (matches MNIST format)")
        print("✅ Enhanced prediction smoothing and confidence handling")
        print("✅ Improved hand detection and bounding box calculation")
        print()
        print("Expected Results:")
        print("• Much more accurate predictions (should fix the 'K' stuck issue)")
        print("• Stable predictions with proper confidence scores")
        print("• Clear debug visualization showing corrected preprocessing")
        print("="*80)
        
        screenshot_count = 0
        top_3_predictions = []
        show_debug_window = True
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Could not read frame")
                    break
                
                # Flip for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Process hand detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Reset prediction tracking
                hand_detected = False
                
                if results.multi_hand_landmarks:
                    hand_detected = True
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract hand region
                        hand_region, bbox = self.extract_hand_region_robust(frame, hand_landmarks)
                        
                        # Draw enhanced hand detection
                        self.draw_enhanced_hand_detection(frame, hand_landmarks, bbox)
                        
                        # CORRECTED preprocessing with inversion fix
                        model_input, processed_debug = self.corrected_image_preprocessing(hand_region)
                        
                        if model_input is not None:
                            # Get enhanced prediction
                            result = self.predict_sign_enhanced(model_input)
                            
                            if len(result) == 3:
                                prediction, confidence, top_3_predictions = result
                                
                                if prediction:
                                    self.current_prediction = prediction
                                    self.current_confidence = confidence
                                    
                                    # Enhanced smoothing
                                    self.stable_prediction = self.smooth_predictions_enhanced(
                                        prediction, confidence
                                    )
                
                # Clear predictions if no hand detected
                if not hand_detected:
                    self.current_prediction = ""
                    self.current_confidence = 0.0
                    self.processed_image_for_debug = None
                    top_3_predictions = []
                
                # Draw final UI
                self.draw_final_prediction_ui(frame, top_3_predictions)
                
                # Create enhanced debug window
                if show_debug_window:
                    self.create_enhanced_debug_window()
                
                # Calculate FPS
                self.calculate_fps()
                
                # Display main frame
                cv2.imshow('FINAL ASL Recognition (FIXED)', frame)
                
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
                elif key == ord('t'):
                    # Toggle inversion method
                    self.use_inverted_threshold = not self.use_inverted_threshold
                    threshold_type = "THRESH_BINARY_INV" if self.use_inverted_threshold else "THRESH_BINARY"
                    print(f"Switched to: {threshold_type}")
                elif key == ord('d'):
                    # Toggle debug window
                    show_debug_window = not show_debug_window
                    if not show_debug_window:
                        cv2.destroyWindow('CORRECTED Model Input Debug')
                elif key == ord('r'):
                    # Reset predictions
                    self.prediction_buffer.clear()
                    self.stable_prediction = ""
                    self.current_prediction = ""
                    print("Predictions reset")
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_count += 1
                    screenshot_path = f"final_screenshot_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\nStopping recognition...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n🎯 Final ASL Recognition ended")
            print(f"Total frames processed: {self.frame_count}")
            if hasattr(self, 'fps') and self.fps > 0:
                print(f"Average FPS: {self.fps:.1f}")
            print("Inversion fix should have resolved the 'K' prediction issue!")

def main():
    """Main application entry point"""
    print("🎯 Final ASL Recognition System")
    print("Complete solution with inversion fix for MNIST compatibility")
    print("="*70)
    
    # Check for trained models
    models_dir = "trained_models"
    if not os.path.exists(models_dir):
        print("❌ No models directory found!")
        print("Please run 'train_sign_mnist_model.py' first to train a model")
        return
    
    model_files = glob.glob(os.path.join(models_dir, "*.h5"))
    if not model_files:
        print("❌ No trained models found!")
        print("Please run 'train_sign_mnist_model.py' first to train a model")
        return
    
    # Initialize final recognizer
    recognizer = FinalASLRecognizer(models_dir)
    
    if recognizer.model:
        print("✅ System ready with INVERSION FIX!")
        print("\nModel Information:")
        if recognizer.metadata:
            print(f"  Test Accuracy: {recognizer.metadata.get('test_accuracy', 'Unknown')}")
        print(f"  Available Classes: {len(recognizer.class_names)}")
        print("\n🔧 CRITICAL FIX Applied:")
        print("  • THRESH_BINARY_INV: Creates white hand on black background")
        print("  • Matches MNIST training data format exactly")
        print("  • Should fix the consistent 'K' prediction issue")
        print("\n💡 If predictions are still wrong, try pressing 'T' to toggle threshold type")
        print()
        
        input("Press Enter to start the CORRECTED recognition system...")
        recognizer.run_final_recognition()
    else:
        print("❌ Failed to load model. Please check your trained models.")

if __name__ == "__main__":
    main()