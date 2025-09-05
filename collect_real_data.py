#!/usr/bin/env python3
"""
Real ASL Data Collection Tool
Collect personalized training data using your webcam
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import mediapipe as mp
from datetime import datetime

class ASLDataCollector:
    def __init__(self, output_dir="real_asl_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # ASL letters (excluding J and Z)
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                       'V', 'W', 'X', 'Y']
        
        self.current_letter_idx = 0
        self.samples_per_letter = 50
        self.collected_count = 0
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Data storage
        self.collected_data = []
        
    def process_hand_region(self, frame, hand_landmarks):
        """Extract and process hand region like the recognition system"""
        h, w, _ = frame.shape
        
        # Get bounding box
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Extract hand region
        hand_roi = frame[y_min:y_max, x_min:x_max]
        
        if hand_roi.size == 0:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        
        # Create hand mask using landmarks
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Get landmark points in ROI coordinates
        points = []
        for lm in hand_landmarks.landmark:
            x = int((lm.x * w - x_min))
            y = int((lm.y * h - y_min))
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                points.append([x, y])
        
        if len(points) > 5:
            # Create convex hull
            points = np.array(points)
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], 255)
            
            # Apply mask
            masked = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Apply threshold
            _, binary = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Resize to 28x28
            resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
            
            return resized
        
        return None
    
    def run_collection(self):
        """Main data collection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return
        
        print("ASL REAL DATA COLLECTION")
        print("=" * 40)
        print("Instructions:")
        print("- Make the displayed letter sign")
        print("- Press SPACE to capture sample")
        print("- Press 'n' for next letter")
        print("- Press 'q' to quit")
        print("=" * 40)
        
        collecting = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)  # Mirror for easier use
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            results = self.hands.process(rgb_frame)
            
            # Current letter info
            if self.current_letter_idx < len(self.letters):
                current_letter = self.letters[self.current_letter_idx]
                progress = f"{self.collected_count}/{self.samples_per_letter}"
            else:
                current_letter = "COMPLETE"
                progress = "ALL DONE!"
            
            # Draw UI
            cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"Letter: {current_letter}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {progress}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "SPACE=Capture, N=Next, Q=Quit", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Process hand detection
            processed_image = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Process for data collection
                    processed_image = self.process_hand_region(frame, hand_landmarks)
                    
                    # Show processed image
                    if processed_image is not None:
                        # Scale up processed image for display
                        display_processed = cv2.resize(processed_image, (140, 140), interpolation=cv2.INTER_NEAREST)
                        frame[130:270, 20:160] = cv2.cvtColor(display_processed, cv2.COLOR_GRAY2BGR)
                        
                        cv2.rectangle(frame, (18, 128), (162, 272), (0, 255, 0), 2)
                        cv2.putText(frame, "Processed", (25, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "No hand detected", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('ASL Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') and processed_image is not None:
                # Capture sample
                if self.current_letter_idx < len(self.letters):
                    self.capture_sample(processed_image, current_letter)
            elif key == ord('n'):
                # Next letter
                self.next_letter()
        
        cap.release()
        cv2.destroyAllWindows()
        self.save_dataset()
    
    def capture_sample(self, processed_image, letter):
        """Capture a training sample"""
        label = self.current_letter_idx
        pixels = processed_image.flatten() / 255.0  # Normalize
        
        self.collected_data.append({
            'label': label,
            'letter': letter,
            'pixels': pixels.tolist(),
            'timestamp': datetime.now().isoformat()
        })
        
        self.collected_count += 1
        print(f"Captured {letter} sample {self.collected_count}/{self.samples_per_letter}")
        
        # Auto-advance if enough samples collected
        if self.collected_count >= self.samples_per_letter:
            self.next_letter()
    
    def next_letter(self):
        """Move to next letter"""
        if self.collected_count > 0:
            self.current_letter_idx += 1
            self.collected_count = 0
            
            if self.current_letter_idx < len(self.letters):
                print(f"\\nNow collecting: {self.letters[self.current_letter_idx]}")
            else:
                print("\\nCollection complete!")
    
    def save_dataset(self):
        """Save collected dataset"""
        if not self.collected_data:
            print("No data collected!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        output_file = self.output_dir / f"real_asl_data_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(self.collected_data, f)
        
        # Convert to CSV format for training
        import pandas as pd
        
        # Prepare data for CSV
        rows = []
        for sample in self.collected_data:
            row = [sample['label']] + sample['pixels']
            rows.append(row)
        
        # Create DataFrame
        columns = ['label'] + [f'pixel{i}' for i in range(784)]
        df = pd.DataFrame(rows, columns=columns)
        
        # Save CSV
        csv_file = self.output_dir / f"real_asl_train_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\\nDataset saved!")
        print(f"Raw data: {output_file}")
        print(f"Training CSV: {csv_file}")
        print(f"Total samples: {len(self.collected_data)}")
        print(f"Letters collected: {len(set(s['letter'] for s in self.collected_data))}")
        
        # Copy to main data directory
        main_data_file = Path("data") / "real_sign_mnist_train.csv"
        Path("data").mkdir(exist_ok=True)
        
        import shutil
        shutil.copy2(csv_file, main_data_file)
        print(f"Copied to: {main_data_file}")

def main():
    collector = ASLDataCollector()
    collector.run_collection()

if __name__ == "__main__":
    main()