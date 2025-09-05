#!/usr/bin/env python3
"""
ASL Recognition Diagnostic Tool
Provides side-by-side comparison between live camera preprocessing and MNIST training data
to identify data mismatch issues (especially inversion problems)
"""

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import mediapipe as mp
import json
import os
import glob
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ASLDiagnosticTool:
    def __init__(self, models_dir="trained_models", data_path="data/sign_mnist_train.csv"):
        self.models_dir = models_dir
        self.data_path = data_path
        
        # Initialize MediaPipe Hand Detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Model components
        self.model = None
        self.class_mapping = None
        self.class_names = []
        
        # MNIST data for comparison
        self.mnist_data = None
        self.current_reference_image = None
        self.current_reference_letter = 'K'  # Default to K since that's what it's predicting
        
        # Prediction settings
        self.confidence_threshold = 0.6
        
        # Processing parameters
        self.blur_kernel_size = 5
        self.adaptive_thresh_block_size = 11
        self.adaptive_thresh_c = 2
        self.hand_padding = 50
        
        # Debug state
        self.processed_live_image = None
        self.is_paused = False
        self.paused_frame = None
        self.paused_processed = None
        
        # Load everything
        self.load_model()
        self.load_mnist_data()
        self.load_reference_image()
        
        print("ASL Diagnostic Tool Initialized")
        print(f"Model loaded: {self.model is not None}")
        print(f"MNIST data loaded: {self.mnist_data is not None}")
        print(f"Reference letter: {self.current_reference_letter}")
    
    def load_model(self):
        """Load the trained model and class mapping"""
        # Find latest model (same as robust recognizer)
        model_files = glob.glob(os.path.join(self.models_dir, "*.h5"))
        if not model_files:
            print("❌ No trained models found!")
            return False
        
        try:
            # Load model
            latest_model = max(model_files, key=os.path.getctime)
            self.model = tf.keras.models.load_model(latest_model)
            print(f"✅ Model loaded: {os.path.basename(latest_model)}")
            
            # Load class mapping
            mapping_files = glob.glob(os.path.join(self.models_dir, "*class_mapping*.json"))
            if mapping_files:
                latest_mapping = max(mapping_files, key=os.path.getctime)
                with open(latest_mapping, 'r') as f:
                    self.class_mapping = json.load(f)
                self.class_names = self.class_mapping['classes']
                print(f"✅ Classes loaded: {len(self.class_names)}")
            else:
                # Fallback
                self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                                  'V', 'W', 'X', 'Y']
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_mnist_data(self):
        """Load MNIST training data for comparison"""
        if not os.path.exists(self.data_path):
            print(f"❌ MNIST data file not found: {self.data_path}")
            return False
        
        try:
            print(f"Loading MNIST data from: {self.data_path}")
            self.mnist_data = pd.read_csv(self.data_path)
            print(f"✅ MNIST data loaded: {self.mnist_data.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading MNIST data: {e}")
            return False
    
    def load_reference_image(self, letter='K'):
        """Load a reference image for the specified letter from MNIST data"""
        if self.mnist_data is None:
            return None
        
        try:
            # Get class index for the letter
            letter_index = self.class_names.index(letter) if letter in self.class_names else 10  # Default to K
            
            # Find samples of this letter
            letter_samples = self.mnist_data[self.mnist_data['label'] == letter_index]
            
            if len(letter_samples) > 0:
                # Get a random sample
                sample_row = letter_samples.sample(n=1).iloc[0]
                
                # Extract pixel values and reshape to 28x28
                pixel_values = sample_row.iloc[1:].values  # Skip label column
                reference_image = pixel_values.reshape(28, 28).astype(np.uint8)
                
                self.current_reference_image = reference_image
                self.current_reference_letter = letter
                
                print(f"✅ Reference image loaded for letter '{letter}'")
                print(f"   Pixel range: {reference_image.min()}-{reference_image.max()}")
                return reference_image
            else:
                print(f"❌ No samples found for letter '{letter}'")
                return None
                
        except Exception as e:
            print(f"❌ Error loading reference image: {e}")
            return None
    
    def extract_hand_region(self, frame, hand_landmarks):
        """Extract hand region (same as robust recognizer)"""
        h, w, c = frame.shape
        
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        x_pixels = [int(x * w) for x in x_coords]
        y_pixels = [int(y * h) for y in y_coords]
        
        padding = self.hand_padding
        x_min = max(0, min(x_pixels) - padding)
        x_max = min(w, max(x_pixels) + padding)
        y_min = max(0, min(y_pixels) - padding)
        y_max = min(h, max(y_pixels) + padding)
        
        # Ensure square aspect ratio
        box_width = x_max - x_min
        box_height = y_max - y_min
        max_side = max(box_width, box_height)
        
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        half_side = max_side // 2
        x_min = max(0, center_x - half_side)
        x_max = min(w, center_x + half_side)
        y_min = max(0, center_y - half_side)
        y_max = min(h, center_y + half_side)
        
        hand_region = frame[y_min:y_max, x_min:x_max]
        return hand_region, (x_min, y_min, x_max, y_max)
    
    def preprocess_live_image(self, hand_region):
        """Preprocess live image (same as robust recognizer)"""
        if hand_region.size == 0:
            return None
        
        # Convert to grayscale
        if len(hand_region.shape) == 3:
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = hand_region.copy()
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,  # This might be the issue!
            self.adaptive_thresh_block_size,
            self.adaptive_thresh_c
        )
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Resize to 28x28
        resized = cv2.resize(cleaned, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Store for debug visualization
        self.processed_live_image = resized
        
        return resized
    
    def predict_current_image(self, processed_image):
        """Make prediction on processed image"""
        if self.model is None or processed_image is None:
            return None, 0.0
        
        try:
            # Normalize and reshape for model
            normalized = processed_image.astype('float32') / 255.0
            model_input = normalized.reshape(1, 28, 28, 1)
            
            # Predict
            predictions = self.model.predict(model_input, verbose=0)
            class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][class_index])
            
            if class_index < len(self.class_names):
                return self.class_names[class_index], confidence
            else:
                return "Unknown", confidence
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def create_debug_comparison_window(self):
        """Create side-by-side comparison window"""
        if self.processed_live_image is None or self.current_reference_image is None:
            # Show placeholder
            placeholder = np.zeros((400, 700, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No comparison available", (200, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Debug View - Data Comparison', placeholder)
            return
        
        # Scale both images to 300x300 for visibility
        debug_size = 300
        
        # Left side: Live processed image
        live_scaled = cv2.resize(self.processed_live_image, (debug_size, debug_size), 
                                interpolation=cv2.INTER_NEAREST)
        live_bgr = cv2.cvtColor(live_scaled, cv2.COLOR_GRAY2BGR)
        
        # Right side: MNIST reference image
        ref_scaled = cv2.resize(self.current_reference_image, (debug_size, debug_size), 
                               interpolation=cv2.INTER_NEAREST)
        ref_bgr = cv2.cvtColor(ref_scaled, cv2.COLOR_GRAY2BGR)
        
        # Create comparison image
        comparison = np.hstack([live_bgr, ref_bgr])
        
        # Add separating line
        cv2.line(comparison, (debug_size, 0), (debug_size, debug_size), (0, 255, 0), 3)
        
        # Add labels and information
        info_height = 150
        info_panel = np.zeros((info_height, debug_size * 2, 3), dtype=np.uint8)
        
        # Labels
        cv2.putText(info_panel, "LIVE CAMERA (Processed)", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, f"MNIST REFERENCE ('{self.current_reference_letter}')", (debug_size + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Pixel statistics
        live_stats = f"Min: {self.processed_live_image.min()}, Max: {self.processed_live_image.max()}"
        ref_stats = f"Min: {self.current_reference_image.min()}, Max: {self.current_reference_image.max()}"
        
        cv2.putText(info_panel, live_stats, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(info_panel, ref_stats, (debug_size + 20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Color coding explanation
        cv2.putText(info_panel, "White = Foreground, Black = Background", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Instructions
        instructions = [
            "Press 'P' to pause/unpause",
            "Press 'R' to change reference letter",
            "Press 'I' to invert live image test",
            "Look for inversion mismatch!"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(info_panel, instruction, (20, 110 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Combine everything
        final_debug = np.vstack([info_panel, comparison])
        
        # Add pause indicator
        if self.is_paused:
            cv2.putText(final_debug, "PAUSED", (debug_size - 50, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        cv2.imshow('Debug View - Data Comparison', final_debug)
    
    def draw_main_ui(self, frame, prediction, confidence):
        """Draw main camera UI"""
        h, w = frame.shape[:2]
        
        # Main info panel
        cv2.rectangle(frame, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 120), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "ASL Diagnostic Tool", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current prediction
        if prediction:
            pred_color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 255, 255)
            cv2.putText(frame, f"Prediction: {prediction}", (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 1)
        else:
            cv2.putText(frame, "No prediction", (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Status
        status_y = h - 50
        cv2.rectangle(frame, (10, status_y - 5), (w - 10, h - 10), (0, 0, 0), -1)
        
        status_text = f"Reference: {self.current_reference_letter} | "
        status_text += "PAUSED" if self.is_paused else "RUNNING"
        cv2.putText(frame, status_text, (20, status_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Controls
        controls = [
            "Controls:",
            "P - Pause/Unpause",
            "R - Change reference",
            "I - Test inversion",
            "Q - Quit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (w - 200, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    def change_reference_letter(self):
        """Allow user to change reference letter"""
        print(f"\nCurrent reference letter: {self.current_reference_letter}")
        print("Available letters:", ', '.join(self.class_names))
        try:
            new_letter = input("Enter new reference letter: ").upper().strip()
            if new_letter in self.class_names:
                self.load_reference_image(new_letter)
                print(f"Reference changed to: {new_letter}")
            else:
                print("Invalid letter. Please choose from available letters.")
        except KeyboardInterrupt:
            print("Reference change cancelled.")
    
    def test_inversion(self):
        """Test what happens if we invert the live image"""
        if self.processed_live_image is not None:
            # Create inverted version
            inverted = 255 - self.processed_live_image
            
            # Make prediction on inverted image
            prediction, confidence = self.predict_current_image(inverted)
            
            print(f"\n=== INVERSION TEST ===")
            print(f"Original prediction: {self.predict_current_image(self.processed_live_image)}")
            print(f"Inverted prediction: {prediction} (confidence: {confidence:.3f})")
            print("======================")
            
            # Temporarily show inverted version in debug window
            temp_original = self.processed_live_image.copy()
            self.processed_live_image = inverted
            self.create_debug_comparison_window()
            cv2.waitKey(2000)  # Show for 2 seconds
            self.processed_live_image = temp_original
    
    def run_diagnostic(self):
        """Main diagnostic loop"""
        if not self.model or self.mnist_data is None:
            print("❌ Cannot run diagnostic. Model or MNIST data not loaded.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*80)
        print("🔍 ASL DIAGNOSTIC TOOL - DATA MISMATCH DETECTOR")
        print("="*80)
        print("This tool helps identify inversion and preprocessing issues")
        print()
        print("Instructions:")
        print("1. Make an ASL sign (try 'K' first since that's what it predicts)")
        print("2. Press 'P' to pause and examine the Debug View window")
        print("3. Compare Live vs MNIST images side-by-side")
        print("4. Press 'I' to test image inversion")
        print("5. Look for white/black inversion between the two images")
        print("="*80)
        
        current_prediction = None
        current_confidence = 0.0
        
        try:
            while True:
                if not self.is_paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    self.paused_frame = frame.copy()  # Store for paused mode
                else:
                    frame = self.paused_frame.copy()  # Use stored frame when paused
                
                # Process hand detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks and not self.is_paused:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                                  self.mp_hands.HAND_CONNECTIONS)
                        
                        # Extract and process hand
                        hand_region, bbox = self.extract_hand_region(frame, hand_landmarks)
                        processed = self.preprocess_live_image(hand_region)
                        
                        if processed is not None:
                            # Make prediction
                            current_prediction, current_confidence = self.predict_current_image(processed)
                            
                            # Draw bounding box
                            if bbox:
                                x_min, y_min, x_max, y_max = bbox
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                elif self.is_paused and hasattr(self, 'paused_processed') and self.paused_processed is not None:
                    # Use stored processed image when paused
                    self.processed_live_image = self.paused_processed
                
                # Draw UI
                self.draw_main_ui(frame, current_prediction, current_confidence)
                
                # Create debug comparison window
                self.create_debug_comparison_window()
                
                # Display main frame
                cv2.imshow('Main Camera View', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    # Toggle pause
                    self.is_paused = not self.is_paused
                    if self.is_paused:
                        self.paused_processed = self.processed_live_image.copy() if self.processed_live_image is not None else None
                        print("PAUSED - Examine the Debug View window")
                    else:
                        print("RESUMED")
                elif key == ord('r'):
                    # Change reference letter
                    self.change_reference_letter()
                elif key == ord('i'):
                    # Test inversion
                    self.test_inversion()
        
        except KeyboardInterrupt:
            print("\nStopping diagnostic...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main diagnostic function"""
    print("🔍 ASL Recognition Diagnostic Tool")
    print("Detects data mismatch issues between live camera and MNIST training data")
    print()
    
    # Check requirements
    if not os.path.exists("trained_models"):
        print("❌ No trained_models directory found!")
        return
    
    if not os.path.exists("data/sign_mnist_train.csv"):
        print("❌ MNIST training data not found!")
        print("Please ensure data/sign_mnist_train.csv exists")
        return
    
    # Initialize and run diagnostic
    diagnostic = ASLDiagnosticTool()
    
    if diagnostic.model and diagnostic.mnist_data is not None:
        print("✅ Diagnostic tool ready!")
        input("Press Enter to start diagnostic mode...")
        diagnostic.run_diagnostic()
    else:
        print("❌ Failed to initialize diagnostic tool")

if __name__ == "__main__":
    main()