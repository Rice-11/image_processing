#!/usr/bin/env python3
"""
ASL Landmark Data Collector
Collects normalized 3D hand landmark data for training
"""

import cv2
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time

class LandmarkDataCollector:
    def __init__(self, output_file="asl_landmark_data.json"):
        self.output_file = Path(output_file)
        self.collected_data = defaultdict(list)
        
        # MediaPipe setup for high-quality landmark detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.9,  # High confidence for quality data
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        # ASL letters (excluding J and Z which require motion)
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                       'V', 'W', 'X', 'Y']
        
        # Collection settings
        self.samples_per_letter = 100
        self.collection_rate = 10  # Collect every 10th frame for diversity
        
        # UI state
        self.current_letter = None
        self.is_collecting = False
        self.frame_counter = 0
        self.last_collection_time = 0
        
        # Load existing data if available
        self.load_existing_data()
        
        print("ASL Landmark Data Collector Initialized")
        print("=" * 50)
        print("Hand landmark normalization process:")
        print("1. Translate: Move wrist (landmark 0) to origin")
        print("2. Scale: Normalize max distance to 1.0")
        print("3. Flatten: Create 63-element feature vector")
        print("=" * 50)
    
    def normalize_landmarks(self, landmarks):
        """
        Normalize hand landmarks for translation/scale invariance
        
        Args:
            landmarks: MediaPipe landmark list
            
        Returns:
            np.array: 63-element normalized feature vector (21 landmarks × 3 coords)
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
    
    def load_existing_data(self):
        """Load existing collected data if available"""
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert back to defaultdict
                for letter, samples in data.items():
                    self.collected_data[letter] = samples
                    
                print(f"Loaded existing data: {sum(len(samples) for samples in self.collected_data.values())} samples")
                
            except Exception as e:
                print(f"Warning: Could not load existing data: {e}")
    
    def save_data(self):
        """Save collected data to JSON file"""
        # Add metadata
        data_with_metadata = {
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'normalization_method': 'wrist_centered_unit_scale',
                'feature_dimension': 63,  # 21 landmarks × 3 coordinates
                'landmark_names': [
                    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
                    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
                    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
                    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
                ],
                'total_samples': sum(len(samples) for samples in self.collected_data.values()),
                'letters_collected': list(self.collected_data.keys())
            },
            'data': dict(self.collected_data)  # Convert defaultdict to regular dict
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(data_with_metadata, f, indent=2)
        
        print(f"\\nData saved to: {self.output_file}")
        print(f"Total samples: {data_with_metadata['metadata']['total_samples']}")
    
    def draw_ui(self, frame):
        """Draw collection UI on frame"""
        h, w, _ = frame.shape
        
        # Background panel
        cv2.rectangle(frame, (10, 10), (w-10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (w-8, 122), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "ASL LANDMARK DATA COLLECTOR", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Current status
        if self.current_letter:
            collected = len(self.collected_data[self.current_letter])
            status = f"Letter: {self.current_letter} | Collected: {collected}/{self.samples_per_letter}"
            color = (0, 255, 0) if collected < self.samples_per_letter else (255, 0, 255)
            cv2.putText(frame, status, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if self.is_collecting:
                cv2.putText(frame, "COLLECTING... Hold steady!", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Press SPACE to start collecting", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Press a letter key (A-Y) to select", (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Controls: Letter keys=Select | SPACE=Collect | S=Save | Q=Quit", 
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Collection summary
        total_samples = sum(len(samples) for samples in self.collected_data.values())
        letters_done = len([l for l, samples in self.collected_data.items() 
                           if len(samples) >= self.samples_per_letter])
        
        summary = f"Total: {total_samples} samples | Complete letters: {letters_done}/24"
        cv2.putText(frame, summary, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def run_collection(self):
        """Main collection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\\nCollection started. Make sure you have good lighting!")
        print("Hold your hand steady in front of the camera.")
        print("Each letter needs diverse poses - vary your hand angle and position slightly.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror for easier use
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_counter += 1
            
            # Process hand landmarks
            results = self.hands.process(rgb_frame)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Handle hand detection and collection
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw beautiful hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )
                    
                    # Show that hand is detected
                    cv2.putText(frame, "Hand Detected ✓", (frame.shape[1]-150, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Collect data if conditions are met
                    if (self.is_collecting and self.current_letter and 
                        self.frame_counter % self.collection_rate == 0):
                        
                        current_time = time.time()
                        if current_time - self.last_collection_time > 0.1:  # 10 FPS collection rate
                            
                            # Normalize landmarks
                            normalized_features = self.normalize_landmarks(hand_landmarks)
                            
                            # Store sample with timestamp
                            sample = {
                                'features': normalized_features.tolist(),
                                'timestamp': datetime.now().isoformat(),
                                'quality_score': self.calculate_quality_score(hand_landmarks)
                            }
                            
                            self.collected_data[self.current_letter].append(sample)
                            self.last_collection_time = current_time
                            
                            collected = len(self.collected_data[self.current_letter])
                            print(f"Collected {self.current_letter}: {collected}/{self.samples_per_letter}")
                            
                            # Auto-stop collection when done
                            if collected >= self.samples_per_letter:
                                self.is_collecting = False
                                print(f"✓ Completed collecting {self.current_letter}!")
                                
                                # Auto-suggest next letter
                                next_letter = self.suggest_next_letter()
                                if next_letter:
                                    print(f"Suggestion: Press '{next_letter.lower()}' for next letter")
            else:
                cv2.putText(frame, "No hand detected", (frame.shape[1]-150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('ASL Landmark Collector', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_data()
            elif key == ord(' '):
                if self.current_letter:
                    self.is_collecting = not self.is_collecting
                    if self.is_collecting:
                        print(f"Started collecting {self.current_letter}")
                    else:
                        print("Stopped collecting")
            elif chr(key).upper() in self.letters:
                letter = chr(key).upper()
                self.current_letter = letter
                self.is_collecting = False
                collected = len(self.collected_data[letter])
                print(f"Selected letter: {letter} (currently {collected} samples)")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Auto-save on exit
        if self.collected_data:
            self.save_data()
    
    def calculate_quality_score(self, landmarks):
        """Calculate a quality score for the landmark data"""
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Calculate spread of landmarks (more spread = better quality)
        spread = np.std(coords, axis=0).mean()
        
        # Calculate visibility (all landmarks should be visible)
        visibility = np.mean([lm.visibility for lm in landmarks.landmark])
        
        return float(spread * visibility)
    
    def suggest_next_letter(self):
        """Suggest the next letter to collect"""
        for letter in self.letters:
            if len(self.collected_data[letter]) < self.samples_per_letter:
                return letter
        return None

def main():
    print("Starting ASL Landmark Data Collection...")
    print("This will create normalized 3D hand landmark data for training.")
    print("Make sure you have good lighting and a clear background.")
    
    collector = LandmarkDataCollector()
    collector.run_collection()
    
    print("\\nCollection session complete!")
    total_samples = sum(len(samples) for samples in collector.collected_data.values())
    print(f"Total samples collected: {total_samples}")
    
    if total_samples > 0:
        print("\\nNext steps:")
        print("1. Run: python landmark_trainer.py")
        print("2. Then: python landmark_predictor.py")

if __name__ == "__main__":
    main()