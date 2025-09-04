#!/usr/bin/env python3
"""
ASL Data Collection Tool
Interactive tool for collecting hand landmark data for ASL letter recognition
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox
import threading

class ASLDataCollector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # Data collection settings
        self.data_dir = "asl_training_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Collection state
        self.current_letter = None
        self.is_collecting = False
        self.collected_data = defaultdict(list)
        self.collection_count = 0
        self.target_samples = 50
        
        # ASL alphabet (excluding J and Z as they require motion)
        self.asl_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                           'V', 'W', 'X', 'Y']
        
        print("ASL Data Collection Tool Initialized")
        print("Available letters: {}".format(', '.join(self.asl_letters)))
        print("Target samples per letter: {}".format(self.target_samples))
    
    def extract_hand_landmarks(self, hand_landmarks):
        """Extract normalized hand landmarks as feature vector"""
        landmarks = []
        
        # Get wrist position as reference point for normalization
        wrist = hand_landmarks.landmark[0]
        
        # Extract all 21 landmarks relative to wrist
        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x - wrist.x,  # Relative x position
                landmark.y - wrist.y,  # Relative y position  
                landmark.z - wrist.z   # Relative z position
            ])
        
        return np.array(landmarks, dtype=np.float32)
    
    def draw_collection_ui(self, frame):
        """Draw collection interface on frame"""
        # Background for UI
        cv2.rectangle(frame, (10, 10), (600, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (600, 150), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "ASL Data Collection Tool", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current letter
        if self.current_letter:
            status_color = (0, 255, 0) if self.is_collecting else (255, 255, 255)
            cv2.putText(frame, f"Current Letter: {self.current_letter}", (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Collection status
            status_text = "COLLECTING..." if self.is_collecting else "Ready to collect"
            cv2.putText(frame, status_text, (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Sample count
            count_text = f"Samples: {self.collection_count}/{self.target_samples}"
            cv2.putText(frame, count_text, (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Progress bar
            progress_width = int(400 * (self.collection_count / self.target_samples))
            cv2.rectangle(frame, (20, 125), (420, 140), (100, 100, 100), -1)
            if progress_width > 0:
                cv2.rectangle(frame, (20, 125), (20 + progress_width, 140), (0, 255, 0), -1)
        
        # Instructions
        instructions = [
            "Controls:",
            "1-9, A-Y: Select letter",
            "SPACE: Start/Stop collection", 
            "S: Save data",
            "Q: Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (frame.shape[1] - 250, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save_data(self):
        """Save collected data to JSON file"""
        if not any(self.collected_data.values()):
            print("No data to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f"asl_data_{timestamp}.json")
        
        # Convert data to serializable format
        save_data = {}
        total_samples = 0
        
        for letter, samples in self.collected_data.items():
            save_data[letter] = {
                'landmarks': [sample.tolist() for sample in samples],
                'count': len(samples)
            }
            total_samples += len(samples)
        
        # Add metadata
        save_data['metadata'] = {
            'timestamp': timestamp,
            'total_samples': total_samples,
            'letters_collected': list(self.collected_data.keys()),
            'samples_per_letter': {k: len(v) for k, v in self.collected_data.items()}
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n" + "="*50)
        print(f"DATA SAVED SUCCESSFULLY!")
        print(f"File: {filename}")
        print(f"Total samples: {total_samples}")
        for letter, samples in self.collected_data.items():
            print(f"  {letter}: {len(samples)} samples")
        print("="*50)
        
        return filename
    
    def load_existing_data(self):
        """Load existing data if available"""
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        if data_files:
            latest_file = max(data_files, key=lambda x: os.path.getctime(os.path.join(self.data_dir, x)))
            filepath = os.path.join(self.data_dir, latest_file)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Load landmark data
                for letter, letter_data in data.items():
                    if letter != 'metadata' and 'landmarks' in letter_data:
                        landmarks = [np.array(sample) for sample in letter_data['landmarks']]
                        self.collected_data[letter] = landmarks
                
                print(f"Loaded existing data from: {latest_file}")
                for letter, samples in self.collected_data.items():
                    print(f"  {letter}: {len(samples)} samples")
                
            except Exception as e:
                print(f"Error loading existing data: {e}")
    
    def run_collection(self):
        """Main data collection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Load existing data
        self.load_existing_data()
        
        print("\n" + "="*60)
        print("ASL DATA COLLECTION STARTED")
        print("="*60)
        print("Instructions:")
        print("1. Choose a letter by pressing its key (A-Y, excluding J,Z)")
        print("2. Make the ASL sign for that letter")
        print("3. Press SPACE to start/stop collecting samples")
        print("4. Collect 50+ samples per letter for best results")
        print("5. Press 'S' to save data when done")
        print("6. Press 'Q' to quit")
        print("="*60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            
            # Process hand detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style()
                    )
                    
                    # Collect data if currently collecting
                    if self.is_collecting and self.current_letter:
                        landmarks = self.extract_hand_landmarks(hand_landmarks)
                        self.collected_data[self.current_letter].append(landmarks)
                        self.collection_count = len(self.collected_data[self.current_letter])
                        
                        # Auto-stop when target reached
                        if self.collection_count >= self.target_samples:
                            self.is_collecting = False
                            print(f"Target reached for letter '{self.current_letter}'!")
            
            # Update collection count for current letter
            if self.current_letter:
                self.collection_count = len(self.collected_data[self.current_letter])
            
            # Draw UI
            self.draw_collection_ui(frame)
            
            # Hand detection status
            if not hand_detected:
                cv2.putText(frame, "No hand detected - Place hand in view", 
                           (50, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            
            cv2.imshow('ASL Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to start/stop collection
                if self.current_letter and hand_detected:
                    self.is_collecting = not self.is_collecting
                    status = "started" if self.is_collecting else "stopped"
                    print(f"Collection {status} for letter '{self.current_letter}'")
                elif not self.current_letter:
                    print("Please select a letter first!")
                elif not hand_detected:
                    print("Please place your hand in view!")
            elif key == ord('s'):  # Save data
                self.save_data()
            elif key >= ord('A') and key <= ord('Z'):  # Letter selection
                letter = chr(key)
                if letter in self.asl_letters:
                    self.current_letter = letter
                    self.is_collecting = False
                    self.collection_count = len(self.collected_data[letter])
                    print(f"Selected letter: {letter}")
                    print(f"Current samples for '{letter}': {self.collection_count}")
                else:
                    print(f"Letter {letter} not available (J and Z require motion)")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final save prompt
        if any(self.collected_data.values()):
            save_final = input("\nSave collected data before exit? (y/n): ")
            if save_final.lower() == 'y':
                self.save_data()

if __name__ == "__main__":
    collector = ASLDataCollector()
    collector.run_collection()