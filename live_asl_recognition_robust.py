#!/usr/bin/env python3
"""
Robust Live ASL Recognition with Advanced Image Processing
Fixes the domain gap between live camera and Sign Language MNIST training data
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

class RobustASLRecognizer:
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
        self.confidence_threshold = 0.6
        self.prediction_buffer = deque(maxlen=12)  # For prediction smoothing
        
        # Display settings
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.stable_prediction = ""
        
        # Image processing parameters
        self.blur_kernel_size = 5
        self.adaptive_thresh_block_size = 11
        self.adaptive_thresh_c = 2
        self.hand_padding = 50  # Increased padding for better hand capture
        
        # Debug visualization
        self.processed_image_for_debug = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.fps_start_time = datetime.now()
        
        # Load the trained model
        self.load_model()
        
        print("Robust ASL Recognizer Initialized")
        print(f"Model loaded: {self.model is not None}")
        print(f"Classes available: {len(self.class_names)}")
        print("Advanced image processing pipeline enabled")
    
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
    
    def advanced_image_preprocessing(self, hand_region):
        """
        Advanced preprocessing to match MNIST format:
        Complex background → Clean black/white hand silhouette
        """
        if hand_region.size == 0:
            return None, None
        
        original_region = hand_region.copy()
        
        # Step 1: Convert to grayscale
        if len(hand_region.shape) == 3:
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = hand_region.copy()
        
        # Step 2: Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Step 3: Adaptive thresholding to create high-contrast black/white image
        # This is the key step to match MNIST format
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred,
            255,  # Max value
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method
            cv2.THRESH_BINARY,  # Threshold type
            self.adaptive_thresh_block_size,  # Block size for threshold calculation
            self.adaptive_thresh_c  # Constant subtracted from mean
        )
        
        # Step 4: Additional morphological operations to clean up the image
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
    
    def predict_sign_robust(self, processed_image):
        """Predict ASL sign with improved confidence handling"""
        if self.model is None or processed_image is None:
            return None, 0.0
        
        try:
            # Get model prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get prediction probabilities
            class_probabilities = predictions[0]
            
            # Get the class with highest probability
            class_index = np.argmax(class_probabilities)
            confidence = float(class_probabilities[class_index])
            
            # Get top 3 predictions for better debugging
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
    
    def smooth_predictions(self, prediction, confidence):
        """Enhanced prediction smoothing"""
        # Only add to buffer if confidence is reasonable
        if confidence >= 0.2:  # Lower threshold for buffer entry
            self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 5:
            return self.stable_prediction
        
        # Get recent high-confidence predictions
        high_conf_predictions = [
            pred for pred, conf in self.prediction_buffer 
            if conf >= self.confidence_threshold
        ]
        
        if not high_conf_predictions:
            # If no high confidence predictions, check for consistent low-confidence ones
            recent_predictions = [pred for pred, conf in list(self.prediction_buffer)[-5:]]
            from collections import Counter
            prediction_counts = Counter(recent_predictions)
            
            if prediction_counts:
                most_common = prediction_counts.most_common(1)[0]
                if most_common[1] >= 3:  # Appears at least 3 times in last 5 predictions
                    return most_common[0]
            
            return ""
        
        # Find most common high-confidence prediction
        from collections import Counter
        prediction_counts = Counter(high_conf_predictions)
        
        if prediction_counts:
            most_common = prediction_counts.most_common(1)[0]
            if most_common[1] >= 2:  # Appears at least twice
                self.stable_prediction = most_common[0]
        
        return self.stable_prediction
    
    def draw_hand_detection_enhanced(self, frame, hand_landmarks, bbox=None):
        """Enhanced hand detection visualization"""
        # Draw hand landmarks with thicker lines
        self.mp_draw.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=3
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
            
            # Corner markers for better visibility
            corner_size = 20
            for (x, y) in [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]:
                cv2.line(frame, (x-corner_size//2, y), (x+corner_size//2, y), (0, 255, 255), 3)
                cv2.line(frame, (x, y-corner_size//2), (x, y+corner_size//2), (0, 255, 255), 3)
            
            # Label
            cv2.putText(frame, "Hand Region", (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def draw_prediction_ui_enhanced(self, frame, top_3_predictions=None):
        """Enhanced prediction UI with debugging info"""
        h, w = frame.shape[:2]
        
        # Main prediction panel
        panel_height = 160
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "Robust ASL Recognition", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Main prediction
        if self.stable_prediction and self.current_confidence >= self.confidence_threshold:
            # Stable prediction (green)
            cv2.putText(frame, f"Letter: {self.stable_prediction}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            
            # Confidence with color coding
            conf_color = (0, 255, 0) if self.current_confidence >= 0.8 else (0, 255, 255)
            cv2.putText(frame, f"Confidence: {self.current_confidence:.3f}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
            
        elif self.current_prediction:
            # Current prediction (yellow)
            cv2.putText(frame, f"Detecting: {self.current_prediction}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {self.current_confidence:.3f} (Building...)", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Place hand in view", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Top 3 predictions for debugging
        if top_3_predictions and len(top_3_predictions) >= 3:
            cv2.putText(frame, "Top predictions:", (20, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            for i, pred in enumerate(top_3_predictions[:3]):
                pred_text = f"{pred['letter']}: {pred['confidence']:.3f}"
                cv2.putText(frame, pred_text, (20 + i * 120, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Processing parameters info
        param_y = h - 100
        cv2.rectangle(frame, (10, param_y - 5), (w - 10, h - 10), (0, 0, 0), -1)
        
        params_text = [
            f"Blur: {self.blur_kernel_size}x{self.blur_kernel_size}",
            f"AdaptThresh: {self.adaptive_thresh_block_size}, C={self.adaptive_thresh_c}",
            f"ConfThreshold: {self.confidence_threshold}",
            f"Buffer: {len(self.prediction_buffer)}/12"
        ]
        
        for i, param in enumerate(params_text):
            cv2.putText(frame, param, (20, param_y + 15 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Performance info
        if hasattr(self, 'fps') and self.fps > 0:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 100, param_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        controls = [
            "Controls:",
            "Q - Quit", "C - Confidence", "B - Blur",
            "T - Threshold", "R - Reset", "S - Screenshot"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (w - 200, 30 + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    
    def create_debug_window(self):
        """Create and update debug visualization window"""
        if self.processed_image_for_debug is not None:
            # Scale up the 28x28 image for better visibility
            debug_size = 300
            debug_image = cv2.resize(
                self.processed_image_for_debug, 
                (debug_size, debug_size), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Convert to BGR for display
            debug_bgr = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
            
            # Add grid lines to show pixel boundaries
            grid_step = debug_size // 28
            for i in range(0, debug_size, grid_step):
                cv2.line(debug_bgr, (i, 0), (i, debug_size), (0, 255, 0), 1)
                cv2.line(debug_bgr, (0, i), (debug_size, i), (0, 255, 0), 1)
            
            # Add information overlay
            info_panel = np.zeros((100, debug_size, 3), dtype=np.uint8)
            cv2.putText(info_panel, "Model Input (28x28)", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_panel, f"Min: {self.processed_image_for_debug.min():.3f}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, f"Max: {self.processed_image_for_debug.max():.3f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, "White=Hand, Black=Background", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Combine debug image with info panel
            combined_debug = np.vstack([info_panel, debug_bgr])
            
            cv2.imshow('Processed Model Input', combined_debug)
        else:
            # Show placeholder
            placeholder = np.zeros((400, 300, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No hand detected", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Processed Model Input', placeholder)
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = datetime.now()
            time_diff = (current_time - self.fps_start_time).total_seconds()
            self.fps = 30 / time_diff
            self.fps_start_time = current_time
    
    def adjust_processing_parameters(self):
        """Interactive parameter adjustment"""
        print(f"\nCurrent processing parameters:")
        print(f"1. Blur kernel size: {self.blur_kernel_size}")
        print(f"2. Adaptive threshold block size: {self.adaptive_thresh_block_size}")
        print(f"3. Adaptive threshold C: {self.adaptive_thresh_c}")
        print(f"4. Confidence threshold: {self.confidence_threshold}")
        print(f"5. Hand padding: {self.hand_padding}")
        
        try:
            param = input("Enter parameter number to adjust (1-5): ").strip()
            
            if param == '1':
                new_val = int(input(f"Enter new blur kernel size (current: {self.blur_kernel_size}): "))
                if new_val > 0 and new_val % 2 == 1:  # Must be odd
                    self.blur_kernel_size = new_val
                    print(f"Blur kernel size set to: {self.blur_kernel_size}")
                else:
                    print("Invalid value. Must be positive and odd.")
                    
            elif param == '2':
                new_val = int(input(f"Enter new block size (current: {self.adaptive_thresh_block_size}): "))
                if new_val > 1 and new_val % 2 == 1:  # Must be > 1 and odd
                    self.adaptive_thresh_block_size = new_val
                    print(f"Adaptive threshold block size set to: {self.adaptive_thresh_block_size}")
                else:
                    print("Invalid value. Must be > 1 and odd.")
                    
            elif param == '3':
                new_val = int(input(f"Enter new C value (current: {self.adaptive_thresh_c}): "))
                self.adaptive_thresh_c = new_val
                print(f"Adaptive threshold C set to: {self.adaptive_thresh_c}")
                
            elif param == '4':
                new_val = float(input(f"Enter new confidence threshold (current: {self.confidence_threshold}): "))
                if 0.1 <= new_val <= 1.0:
                    self.confidence_threshold = new_val
                    print(f"Confidence threshold set to: {self.confidence_threshold}")
                else:
                    print("Invalid value. Must be between 0.1 and 1.0")
                    
            elif param == '5':
                new_val = int(input(f"Enter new hand padding (current: {self.hand_padding}): "))
                if new_val >= 10:
                    self.hand_padding = new_val
                    print(f"Hand padding set to: {self.hand_padding}")
                else:
                    print("Invalid value. Must be >= 10")
                    
            else:
                print("Invalid parameter number.")
                
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("Parameter adjustment cancelled.")
    
    def run_robust_recognition(self):
        """Main recognition loop with enhanced processing"""
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
        print("🚀 ROBUST ASL RECOGNITION WITH ADVANCED PREPROCESSING")
        print("="*80)
        print("New Features:")
        print("• Advanced image preprocessing to match MNIST format")
        print("• Gaussian blur → Adaptive threshold → Morphological cleaning")
        print("• Debug window shows exactly what the model sees")
        print("• Enhanced prediction smoothing and confidence handling")
        print("• Real-time parameter adjustment")
        print()
        print("Instructions:")
        print("• Make clear ASL signs with good contrast against background")
        print("• Watch the 'Processed Model Input' window for debugging")
        print("• White pixels = hand, Black pixels = background (like MNIST)")
        print("• Press 'T' to adjust preprocessing parameters")
        print("="*80)
        
        screenshot_count = 0
        top_3_predictions = []
        
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
                        # Extract hand region with robust method
                        hand_region, bbox = self.extract_hand_region_robust(frame, hand_landmarks)
                        
                        # Draw enhanced hand detection
                        self.draw_hand_detection_enhanced(frame, hand_landmarks, bbox)
                        
                        # Advanced preprocessing
                        model_input, processed_debug = self.advanced_image_preprocessing(hand_region)
                        
                        if model_input is not None:
                            # Get prediction with top-3 results
                            result = self.predict_sign_robust(model_input)
                            
                            if len(result) == 3:  # Unpack enhanced prediction results
                                prediction, confidence, top_3_predictions = result
                                
                                if prediction:
                                    self.current_prediction = prediction
                                    self.current_confidence = confidence
                                    
                                    # Update stable prediction with enhanced smoothing
                                    self.stable_prediction = self.smooth_predictions(prediction, confidence)
                
                # Clear predictions if no hand detected
                if not hand_detected:
                    self.current_prediction = ""
                    self.current_confidence = 0.0
                    self.processed_image_for_debug = None
                    top_3_predictions = []
                
                # Draw enhanced UI
                self.draw_prediction_ui_enhanced(frame, top_3_predictions)
                
                # Create debug visualization window
                self.create_debug_window()
                
                # Calculate FPS
                self.calculate_fps()
                
                # Display main frame
                cv2.imshow('Robust Live ASL Recognition', frame)
                
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
                    # Adjust processing parameters
                    self.adjust_processing_parameters()
                elif key == ord('b'):
                    # Quick blur adjustment
                    self.blur_kernel_size = 3 if self.blur_kernel_size >= 5 else self.blur_kernel_size + 2
                    print(f"Blur kernel size: {self.blur_kernel_size}")
                elif key == ord('r'):
                    # Reset predictions
                    self.prediction_buffer.clear()
                    self.stable_prediction = ""
                    self.current_prediction = ""
                    print("Predictions reset")
                elif key == ord('s'):
                    # Save screenshot with debug info
                    screenshot_count += 1
                    screenshot_path = f"robust_screenshot_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\nStopping recognition...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n🎯 Robust ASL Recognition ended")
            print(f"Total frames processed: {self.frame_count}")
            if hasattr(self, 'fps') and self.fps > 0:
                print(f"Average FPS: {self.fps:.1f}")

def main():
    """Main application entry point"""
    print("🤖 Robust Live ASL Recognition")
    print("Advanced image processing to fix domain gap issues")
    print("="*60)
    
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
    
    # Initialize robust recognizer
    recognizer = RobustASLRecognizer(models_dir)
    
    if recognizer.model:
        print("✅ System ready with advanced processing!")
        print("\nModel Information:")
        if recognizer.metadata:
            print(f"  Test Accuracy: {recognizer.metadata.get('test_accuracy', 'Unknown')}")
        print(f"  Available Classes: {len(recognizer.class_names)}")
        print("\n🔧 Advanced Features Enabled:")
        print("  • Gaussian Blur Noise Reduction")
        print("  • Adaptive Thresholding for High Contrast")
        print("  • Morphological Cleaning")
        print("  • Debug Visualization Window")
        print("  • Enhanced Prediction Smoothing")
        print("  • Real-time Parameter Tuning")
        print()
        
        input("Press Enter to start the enhanced camera system...")
        recognizer.run_robust_recognition()
    else:
        print("❌ Failed to load model. Please check your trained models.")

if __name__ == "__main__":
    main()