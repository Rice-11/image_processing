#!/usr/bin/env python3
"""
Smart ASL Recognition with Intelligent Hand Masking
Uses MediaPipe landmarks to create precise hand masks and clean preprocessing
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

class SmartASLRecognizer:
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
        self.prediction_buffer = deque(maxlen=10)
        
        # Display settings
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.stable_prediction = ""
        
        # Smart processing parameters
        self.hand_mask_expansion = 1.3  # How much to expand the hand mask
        self.preprocessing_method = "smart_mask"  # "smart_mask", "contour", "adaptive"
        
        # Debug visualization
        self.processed_image_for_debug = None
        self.hand_mask_debug = None
        self.original_roi_debug = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.fps_start_time = datetime.now()
        
        # Load the trained model
        self.load_model()
        
        print("Smart ASL Recognizer Initialized")
        print(f"Model loaded: {self.model is not None}")
        print(f"Using intelligent hand masking preprocessing")
    
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
            print("ERROR: No trained models found!")
            return False
        
        try:
            print(f"Loading model: {os.path.basename(model_path)}")
            self.model = tf.keras.models.load_model(model_path)
            print("SUCCESS: Model loaded successfully")
            
            # Load class mapping
            if mapping_path and os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.class_mapping = json.load(f)
                self.class_names = self.class_mapping['classes']
                print(f"SUCCESS: Class mapping loaded: {len(self.class_names)} classes")
            else:
                self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                                  'V', 'W', 'X', 'Y']
                print("WARNING: Using default class mapping")
            
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"SUCCESS: Metadata loaded")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Error loading model: {e}")
            return False
    
    def create_precise_hand_mask(self, frame, hand_landmarks):
        """Create a precise hand mask using MediaPipe landmarks"""
        h, w, c = frame.shape
        
        # Create blank mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get landmark points
        landmark_points = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmark_points.append([x, y])
        
        # Create convex hull around hand landmarks
        landmark_points = np.array(landmark_points, dtype=np.int32)
        hull = cv2.convexHull(landmark_points)
        
        # Expand the hull slightly for better coverage
        center = np.mean(hull, axis=0)
        expanded_hull = center + self.hand_mask_expansion * (hull - center)
        expanded_hull = expanded_hull.astype(np.int32)
        
        # Fill the mask
        cv2.fillPoly(mask, [expanded_hull], 255)
        
        # Apply morphological operations to smooth the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask, expanded_hull
    
    def smart_hand_preprocessing(self, frame, hand_landmarks):
        """
        Smart preprocessing using MediaPipe landmarks for precise hand extraction
        """
        try:
            h, w, c = frame.shape
            
            # Method 1: Smart Masking (Recommended)
            if self.preprocessing_method == "smart_mask":
                return self.method_smart_mask(frame, hand_landmarks)
            
            # Method 2: Contour-based
            elif self.preprocessing_method == "contour":
                return self.method_contour_based(frame, hand_landmarks)
            
            # Method 3: Adaptive (Fallback)
            else:
                return self.method_adaptive_fallback(frame, hand_landmarks)
        
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None, None
    
    def method_smart_mask(self, frame, hand_landmarks):
        """Method 1: Smart masking with landmark-based hand isolation"""
        h, w, c = frame.shape
        
        # Create precise hand mask
        hand_mask, hull = self.create_precise_hand_mask(frame, hand_landmarks)
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask to isolate hand
        masked_hand = cv2.bitwise_and(gray, gray, mask=hand_mask)
        
        # Get bounding box of the mask
        x, y, w_box, h_box = cv2.boundingRect(hull)
        
        # Add padding
        padding = 30
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + w_box + padding)
        y2 = min(h, y + h_box + padding)
        
        # Extract the hand region
        hand_roi = masked_hand[y1:y2, x1:x2]
        mask_roi = hand_mask[y1:y2, x1:x2]
        
        # Store for debug
        self.original_roi_debug = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        self.hand_mask_debug = mask_roi
        
        if hand_roi.size == 0:
            return None, None
        
        # Create clean binary image
        # Where mask is active, keep the hand; where not, make background black
        clean_hand = np.zeros_like(hand_roi)
        
        # Use Otsu's thresholding on the hand region
        if cv2.countNonZero(mask_roi) > 50:  # Need enough pixels for reliable thresholding
            # Apply Otsu's thresholding to the entire hand ROI
            _, binary = cv2.threshold(hand_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Only keep parts within the mask
            clean_hand = cv2.bitwise_and(binary, binary, mask=mask_roi)
        else:
            # Fallback: simple thresholding
            _, clean_hand = cv2.threshold(hand_roi, 127, 255, cv2.THRESH_BINARY)
            clean_hand = cv2.bitwise_and(clean_hand, clean_hand, mask=mask_roi)
        
        # Morphological cleaning
        kernel = np.ones((3, 3), np.uint8)
        clean_hand = cv2.morphologyEx(clean_hand, cv2.MORPH_CLOSE, kernel)
        clean_hand = cv2.morphologyEx(clean_hand, cv2.MORPH_OPEN, kernel)
        
        # Resize to 28x28
        resized = cv2.resize(clean_hand, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Store for debug
        self.processed_image_for_debug = resized
        
        # Normalize and reshape for model
        normalized = resized.astype('float32') / 255.0
        model_input = normalized.reshape(1, 28, 28, 1)
        
        return model_input, resized
    
    def method_contour_based(self, frame, hand_landmarks):
        """Method 2: Contour-based hand extraction"""
        h, w, c = frame.shape
        
        # Create hand mask
        hand_mask, hull = self.create_precise_hand_mask(frame, hand_landmarks)
        
        # Convert to grayscale and apply mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=hand_mask)
        
        # Find contours within the masked region
        contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Get the largest contour (should be the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        
        # Extract region and create clean binary image
        hand_roi = masked[y:y+h_box, x:x+w_box]
        
        if hand_roi.size == 0:
            return None, None
        
        # Apply threshold
        _, binary = cv2.threshold(hand_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Resize to 28x28
        resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
        
        self.processed_image_for_debug = resized
        
        # Normalize and reshape
        normalized = resized.astype('float32') / 255.0
        model_input = normalized.reshape(1, 28, 28, 1)
        
        return model_input, resized
    
    def method_adaptive_fallback(self, frame, hand_landmarks):
        """Method 3: Adaptive thresholding fallback (improved version)"""
        h, w, c = frame.shape
        
        # Get landmark bounding box
        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
        
        padding = 40
        x1 = max(0, min(x_coords) - padding)
        x2 = min(w, max(x_coords) + padding)
        y1 = max(0, min(y_coords) - padding)
        y2 = min(h, max(y_coords) + padding)
        
        # Extract region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for noise reduction while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding with better parameters
        binary = cv2.adaptiveThreshold(
            filtered, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            15, 4
        )
        
        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Resize to 28x28
        resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
        
        self.processed_image_for_debug = resized
        
        # Normalize and reshape
        normalized = resized.astype('float32') / 255.0
        model_input = normalized.reshape(1, 28, 28, 1)
        
        return model_input, resized
    
    def predict_sign_smart(self, processed_image):
        """Smart prediction with detailed analysis"""
        if self.model is None or processed_image is None:
            return None, 0.0, []
        
        try:
            # Get model prediction
            predictions = self.model.predict(processed_image, verbose=0)
            class_probabilities = predictions[0]
            
            # Get top 5 predictions
            top_5_indices = np.argsort(class_probabilities)[-5:][::-1]
            top_5_predictions = []
            
            for idx in top_5_indices:
                if idx < len(self.class_names):
                    top_5_predictions.append({
                        'letter': self.class_names[idx],
                        'confidence': float(class_probabilities[idx])
                    })
            
            # Get best prediction
            best_idx = np.argmax(class_probabilities)
            best_confidence = float(class_probabilities[best_idx])
            
            if best_idx < len(self.class_names):
                best_prediction = self.class_names[best_idx]
                return best_prediction, best_confidence, top_5_predictions
            else:
                return None, best_confidence, top_5_predictions
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, []
    
    def smooth_predictions_smart(self, prediction, confidence):
        """Smart prediction smoothing"""
        if confidence >= 0.4:  # Lower threshold for adding to buffer
            self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 5:
            return self.stable_prediction
        
        # Get high confidence predictions
        high_conf = [pred for pred, conf in self.prediction_buffer if conf >= self.confidence_threshold]
        
        if high_conf:
            from collections import Counter
            counts = Counter(high_conf)
            most_common = counts.most_common(1)[0]
            if most_common[1] >= 3:  # Appears at least 3 times
                self.stable_prediction = most_common[0]
        
        return self.stable_prediction
    
    def create_comprehensive_debug_window(self):
        """Create comprehensive debug window showing all processing steps"""
        if self.processed_image_for_debug is None:
            placeholder = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No hand detected", (300, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Smart Processing Debug', placeholder)
            return
        
        # Scale images for visibility
        debug_size = 200
        
        # Processed final image
        final_scaled = cv2.resize(self.processed_image_for_debug, (debug_size, debug_size), 
                                 interpolation=cv2.INTER_NEAREST)
        final_bgr = cv2.cvtColor(final_scaled, cv2.COLOR_GRAY2BGR)
        
        # Original ROI (if available)
        if self.original_roi_debug is not None:
            original_scaled = cv2.resize(self.original_roi_debug, (debug_size, debug_size))
            original_bgr = cv2.cvtColor(original_scaled, cv2.COLOR_GRAY2BGR)
        else:
            original_bgr = np.zeros((debug_size, debug_size, 3), dtype=np.uint8)
        
        # Hand mask (if available)
        if self.hand_mask_debug is not None:
            mask_scaled = cv2.resize(self.hand_mask_debug, (debug_size, debug_size))
            mask_bgr = cv2.cvtColor(mask_scaled, cv2.COLOR_GRAY2BGR)
        else:
            mask_bgr = np.zeros((debug_size, debug_size, 3), dtype=np.uint8)
        
        # Create comparison layout
        top_row = np.hstack([original_bgr, mask_bgr])
        bottom_row = np.hstack([final_bgr, np.zeros((debug_size, debug_size, 3), dtype=np.uint8)])
        
        debug_image = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_image, "1. Original ROI", (10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_image, "2. Hand Mask", (debug_size + 10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_image, "3. Final 28x28", (10, debug_size + 40), font, 0.5, (255, 255, 255), 1)
        
        # Add method info
        method_text = f"Method: {self.preprocessing_method}"
        cv2.putText(debug_image, method_text, (debug_size + 10, debug_size + 60), font, 0.5, (0, 255, 255), 1)
        
        # Add statistics
        stats_text = f"Min: {self.processed_image_for_debug.min()}, Max: {self.processed_image_for_debug.max()}"
        cv2.putText(debug_image, stats_text, (debug_size + 10, debug_size + 80), font, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Smart Processing Debug', debug_image)
    
    def draw_smart_ui(self, frame, top_5_predictions=None):
        """Draw smart UI with comprehensive information"""
        h, w = frame.shape[:2]
        
        # Main prediction panel
        panel_height = 200
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, panel_height), (255, 255, 255), 3)
        
        # Title
        cv2.putText(frame, "SMART ASL Recognition (Landmark Masking)", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Current method
        method_text = f"Method: {self.preprocessing_method.replace('_', ' ').title()}"
        cv2.putText(frame, method_text, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Main prediction
        if self.stable_prediction and self.current_confidence >= self.confidence_threshold:
            cv2.putText(frame, f"Letter: {self.stable_prediction}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Confidence: {self.current_confidence:.3f}", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.current_prediction:
            cv2.putText(frame, f"Detecting: {self.current_prediction}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {self.current_confidence:.3f}", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Place hand clearly in view", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Top 5 predictions
        if top_5_predictions and len(top_5_predictions) >= 5:
            cv2.putText(frame, "Top 5 predictions:", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            for i, pred in enumerate(top_5_predictions[:5]):
                pred_text = f"{pred['letter']}: {pred['confidence']:.3f}"
                cv2.putText(frame, pred_text, (20 + i * 100, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Controls
        controls = [
            "Controls:",
            "Q - Quit", "M - Change method", "C - Confidence",
            "R - Reset", "D - Debug window", "S - Screenshot"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (w - 220, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Status info
        status_y = h - 40
        cv2.rectangle(frame, (10, status_y - 5), (w - 10, h - 10), (0, 0, 0), -1)
        
        # FPS and method info
        if hasattr(self, 'fps') and self.fps > 0:
            cv2.putText(frame, f"FPS: {self.fps:.1f} | Expansion: {self.hand_mask_expansion:.1f}", 
                       (20, status_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def calculate_fps(self):
        """Calculate FPS"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = datetime.now()
            time_diff = (current_time - self.fps_start_time).total_seconds()
            self.fps = 30 / time_diff
            self.fps_start_time = current_time
    
    def run_smart_recognition(self):
        """Main smart recognition loop"""
        if not self.model:
            print("ERROR: No model loaded. Please train a model first.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*80)
        print("SMART ASL RECOGNITION - INTELLIGENT HAND MASKING")
        print("="*80)
        print("New Approach:")
        print("+ Uses MediaPipe landmarks to create precise hand masks")
        print("+ Multiple preprocessing methods available")
        print("+ Smart binary conversion with Otsu's thresholding")
        print("+ Comprehensive debug visualization")
        print("+ Should eliminate the noise issue completely")
        print("="*80)
        
        show_debug = True
        top_5_predictions = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # MediaPipe hand detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw_styles.get_default_hand_landmarks_style(),
                            self.mp_draw_styles.get_default_hand_connections_style()
                        )
                        
                        # Smart preprocessing
                        model_input, processed_debug = self.smart_hand_preprocessing(frame, hand_landmarks)
                        
                        if model_input is not None:
                            # Smart prediction
                            result = self.predict_sign_smart(model_input)
                            
                            if len(result) == 3:
                                prediction, confidence, top_5_predictions = result
                                
                                if prediction:
                                    self.current_prediction = prediction
                                    self.current_confidence = confidence
                                    self.stable_prediction = self.smooth_predictions_smart(prediction, confidence)
                else:
                    # No hand detected
                    self.current_prediction = ""
                    self.current_confidence = 0.0
                    self.processed_image_for_debug = None
                    top_5_predictions = []
                
                # Draw UI
                self.draw_smart_ui(frame, top_5_predictions)
                
                # Debug window
                if show_debug:
                    self.create_comprehensive_debug_window()
                
                # Calculate FPS
                self.calculate_fps()
                
                # Show main frame
                cv2.imshow('Smart ASL Recognition', frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    # Change preprocessing method
                    methods = ["smart_mask", "contour", "adaptive"]
                    current_idx = methods.index(self.preprocessing_method)
                    self.preprocessing_method = methods[(current_idx + 1) % len(methods)]
                    print(f"Switched to method: {self.preprocessing_method}")
                elif key == ord('c'):
                    # Adjust confidence
                    print(f"Current confidence threshold: {self.confidence_threshold}")
                    try:
                        new_threshold = float(input("Enter new threshold (0.1-1.0): "))
                        if 0.1 <= new_threshold <= 1.0:
                            self.confidence_threshold = new_threshold
                            print(f"Confidence threshold set to: {self.confidence_threshold}")
                    except:
                        print("Invalid input")
                elif key == ord('d'):
                    # Toggle debug window
                    show_debug = not show_debug
                    if not show_debug:
                        cv2.destroyWindow('Smart Processing Debug')
                elif key == ord('r'):
                    # Reset predictions
                    self.prediction_buffer.clear()
                    self.stable_prediction = ""
                    print("Predictions reset")
        
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    print("Smart ASL Recognition System")
    print("Using intelligent MediaPipe landmark masking for precise hand extraction")
    print()
    
    # Check for models
    models_dir = "trained_models"
    if not os.path.exists(models_dir) or not glob.glob(os.path.join(models_dir, "*.h5")):
        print("ERROR: No trained models found!")
        print("Please run 'train_sign_mnist_model.py' first")
        return
    
    recognizer = SmartASLRecognizer(models_dir)
    
    if recognizer.model:
        print("SUCCESS: Smart system ready!")
        print("\nKey Features:")
        print("- Precise hand masking using MediaPipe landmarks")
        print("- Multiple preprocessing methods")
        print("- Comprehensive debug visualization")
        print("- Should solve the noise/quality issues")
        print()
        
        input("Press Enter to start smart recognition...")
        recognizer.run_smart_recognition()
    else:
        print("ERROR: Failed to load model")

if __name__ == "__main__":
    main()