#!/usr/bin/env python3
"""
ASL Live Landmark Predictor
Real-time ASL recognition using 3D hand landmarks
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import pickle
from pathlib import Path
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for Keras models
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class LiveLandmarkPredictor:
    def __init__(self, model_dir="landmark_models"):
        self.model_dir = Path(model_dir)
        
        # Model components
        self.model = None
        self.label_encoder = None
        self.model_type = None
        self.classes = []
        self.model_metadata = {}
        
        # MediaPipe setup - EXACT same configuration as collector
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.9,  # Same as collector
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=10)  # Last 10 predictions
        self.confidence_buffer = deque(maxlen=10)  # Last 10 confidence scores
        self.min_confidence_threshold = 0.6
        self.stability_threshold = 0.7  # Minimum agreement for stable prediction
        
        # Display state
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.stable_prediction = ""
        self.is_stable = False
        self.prediction_history = deque(maxlen=50)  # For visualization
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Load model and components
        self.load_model_components()
        
        print("ASL Live Landmark Predictor Initialized")
        print("=" * 50)
        print("✓ Real-time 3D landmark-based recognition")
        print("✓ Prediction smoothing and confidence filtering")
        print("✓ Exact same normalization as training data")
        print("=" * 50)
    
    def normalize_landmarks(self, landmarks):
        """
        CRITICAL: Exact same normalization as landmark_collector.py
        
        Args:
            landmarks: MediaPipe landmark list
            
        Returns:
            np.array: 63-element normalized feature vector
        """
        # Extract 3D coordinates
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Step 1: Translate - move wrist (landmark 0) to origin
        wrist = coords[0]  # Landmark 0 is always the wrist
        translated = coords - wrist
        
        # Step 2: Scale - normalize by maximum distance from origin
        distances = np.linalg.norm(translated, axis=1)
        max_distance = np.max(distances)
        
        if max_distance > 0:
            normalized = translated / max_distance
        else:
            normalized = translated  # Prevent division by zero
        
        # Step 3: Flatten to 63-element vector
        feature_vector = normalized.flatten()
        
        return feature_vector
    
    def load_model_components(self):
        """Load trained model and supporting components"""
        print("Loading model components...")
        
        # Load metadata
        metadata_path = self.model_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.model_metadata = json.load(f)
        
        self.model_type = self.model_metadata['model_type']
        self.classes = self.model_metadata['classes']
        
        print(f"Model type: {self.model_type}")
        print(f"Classes: {len(self.classes)}")
        print(f"Training accuracy: {self.model_metadata['accuracy']:.4f}")
        
        # Load label encoder
        encoder_path = self.model_dir / "label_encoder.pkl"
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load model based on type
        if self.model_type == 'deep_neural_network':
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow required for deep neural network model")
            
            model_path = self.model_dir / "asl_landmark_model_keras.h5"
            self.model = tf.keras.models.load_model(model_path)
            print(f"✓ Loaded Keras model: {model_path}")
            
        else:
            # Scikit-learn model
            model_path = self.model_dir / "asl_landmark_model_sklearn.pkl"
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Loaded scikit-learn model: {model_path}")
        
        print("✓ All components loaded successfully")
    
    def predict_landmarks(self, landmarks):
        """
        Predict ASL letter from normalized landmarks
        
        Args:
            landmarks: MediaPipe landmark list
            
        Returns:
            tuple: (predicted_letter, confidence_score)
        """
        # Normalize landmarks using EXACT same method as collector
        features = self.normalize_landmarks(landmarks)
        
        # Reshape for model input
        features = features.reshape(1, -1)
        
        # Make prediction based on model type
        if self.model_type == 'deep_neural_network':
            # Keras model returns probabilities
            predictions = self.model.predict(features, verbose=0)
            confidence = float(np.max(predictions))
            predicted_idx = int(np.argmax(predictions))
        else:
            # Scikit-learn model
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)
                confidence = float(np.max(probabilities))
                predicted_idx = int(np.argmax(probabilities))
            else:
                # Model doesn't support probabilities
                predicted_idx = int(self.model.predict(features)[0])
                confidence = 1.0  # Assume high confidence
        
        # Convert to letter
        predicted_letter = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        return predicted_letter, confidence
    
    def update_prediction_smoothing(self, prediction, confidence):
        """Update prediction buffers and compute stable prediction"""
        # Add to buffers
        self.prediction_buffer.append(prediction)
        self.confidence_buffer.append(confidence)
        
        # Only proceed if we have enough samples
        if len(self.prediction_buffer) < 5:
            return prediction, confidence, False
        
        # Filter by confidence threshold
        valid_predictions = [
            pred for pred, conf in zip(self.prediction_buffer, self.confidence_buffer)
            if conf >= self.min_confidence_threshold
        ]
        
        if not valid_predictions:
            return prediction, confidence, False
        
        # Find most common prediction among recent high-confidence predictions
        from collections import Counter
        prediction_counts = Counter(valid_predictions)
        most_common_pred, count = prediction_counts.most_common(1)[0]
        
        # Check if prediction is stable (appears frequently enough)
        stability_ratio = count / len(valid_predictions)
        is_stable = stability_ratio >= self.stability_threshold
        
        if is_stable:
            # Calculate average confidence for the stable prediction
            stable_confidences = [
                conf for pred, conf in zip(self.prediction_buffer, self.confidence_buffer)
                if pred == most_common_pred and conf >= self.min_confidence_threshold
            ]
            avg_confidence = np.mean(stable_confidences) if stable_confidences else confidence
            
            return most_common_pred, avg_confidence, True
        
        return prediction, confidence, False
    
    def draw_prediction_ui(self, frame):
        """Draw prediction interface on frame"""
        h, w, _ = frame.shape
        
        # Main prediction panel
        panel_height = 140
        cv2.rectangle(frame, (10, 10), (w-10, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (w-8, panel_height+2), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "ASL LANDMARK RECOGNITION", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Current prediction
        if self.stable_prediction and self.is_stable:
            pred_text = f"LETTER: {self.stable_prediction}"
            conf_text = f"CONFIDENCE: {self.current_confidence:.1%}"
            
            # Stable prediction - green
            cv2.putText(frame, pred_text, (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, conf_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "✓ STABLE", (w-120, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        elif self.current_prediction:
            pred_text = f"DETECTING: {self.current_prediction}"
            conf_text = f"CONFIDENCE: {self.current_confidence:.1%}"
            
            # Unstable prediction - yellow
            cv2.putText(frame, pred_text, (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, conf_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "~ DETECTING", (w-140, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        else:
            cv2.putText(frame, "Show your hand to camera", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # System info
        cv2.putText(frame, f"FPS: {self.fps:.1f} | Model: {self.model_type.replace('_', ' ').title()}", 
                   (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Prediction history visualization
        if self.prediction_history:
            self.draw_prediction_history(frame)
    
    def draw_prediction_history(self, frame):
        """Draw recent prediction history"""
        history_y = frame.shape[0] - 60
        history_x_start = 20
        
        # Background
        cv2.rectangle(frame, (15, history_y - 25), (frame.shape[1] - 15, history_y + 15), 
                     (0, 0, 0), -1)
        cv2.putText(frame, "Recent predictions:", (20, history_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show last 15 predictions
        recent_predictions = list(self.prediction_history)[-15:]
        for i, (pred, conf, stable) in enumerate(recent_predictions):
            x = history_x_start + i * 25
            color = (0, 255, 0) if stable else (100, 100, 100)
            cv2.putText(frame, pred, (x, history_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        else:
            self.frame_count += 1
    
    def run_live_prediction(self):
        """Main live prediction loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return
        
        # Set camera properties for consistency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\\n🎥 Starting live ASL recognition...")
        print("Hold your hand in front of the camera and make ASL signs")
        print("Press 'q' to quit, 'r' to reset prediction history")
        print("=" * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror for easier use
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Process hand landmarks
            results = self.hands.process(rgb_frame)
            
            # Reset prediction state
            self.current_prediction = ""
            self.current_confidence = 0.0
            self.is_stable = False
            
            # Handle hand detection and prediction
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks with beautiful styling
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )
                    
                    try:
                        # Make prediction
                        prediction, confidence = self.predict_landmarks(hand_landmarks)
                        
                        # Update smoothing
                        stable_pred, stable_conf, is_stable = self.update_prediction_smoothing(
                            prediction, confidence
                        )
                        
                        # Update display state
                        self.current_prediction = prediction
                        self.current_confidence = confidence
                        
                        if is_stable:
                            self.stable_prediction = stable_pred
                            self.current_confidence = stable_conf
                            self.is_stable = True
                            
                            # Add to history
                            self.prediction_history.append((stable_pred, stable_conf, True))
                        else:
                            # Add unstable prediction to history
                            self.prediction_history.append((prediction, confidence, False))
                    
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        continue
            
            # Draw UI
            self.draw_prediction_ui(frame)
            
            # Show frame
            cv2.imshow('ASL Live Landmark Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset prediction history
                self.prediction_buffer.clear()
                self.confidence_buffer.clear()
                self.prediction_history.clear()
                self.stable_prediction = ""
                print("Prediction history reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\\n✅ Live prediction session ended")
        
        # Show session statistics
        if self.prediction_history:
            stable_predictions = [p for p, c, s in self.prediction_history if s]
            total_predictions = len(self.prediction_history)
            stable_count = len(stable_predictions)
            
            print(f"Session statistics:")
            print(f"Total predictions: {total_predictions}")
            print(f"Stable predictions: {stable_count} ({stable_count/total_predictions*100:.1f}%)")
            
            if stable_predictions:
                from collections import Counter
                most_common = Counter(stable_predictions).most_common(5)
                print(f"Most recognized letters: {most_common}")

def main():
    print("ASL Live Landmark Recognition")
    print("Real-time ASL recognition using 3D hand landmarks")
    print("This approach is robust to lighting and background changes!")
    
    try:
        predictor = LiveLandmarkPredictor()
        predictor.run_live_prediction()
        
    except FileNotFoundError as e:
        print(f"\\n❌ Error: {e}")
        print("\\nPlease ensure you have:")
        print("1. Collected training data: python landmark_collector.py")
        print("2. Trained a model: python landmark_trainer.py")
        
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()