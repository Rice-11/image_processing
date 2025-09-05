#!/usr/bin/env python3
"""
Optimized ASL Data Collection - Webcam Specific
Addresses common accuracy issues with personalized data collection
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import mediapipe as mp
from datetime import datetime

class OptimizedASLCollector:
    def __init__(self, output_dir="optimized_asl_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # ASL letters
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                       'V', 'W', 'X', 'Y']
        
        self.current_letter_idx = 0
        self.samples_per_letter = 100  # More samples for better accuracy
        self.collected_count = 0
        
        # Enhanced MediaPipe setup - matching your recognition system
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,  # Higher confidence
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Data storage with variations
        self.collected_data = []
        
        # Quality control
        self.min_hand_area = 2000  # Minimum hand size
        self.max_hand_area = 50000  # Maximum hand size
        
    def process_hand_region_exact_match(self, frame, hand_landmarks):
        """Process hand region EXACTLY like smart_asl_recognition.py"""
        h, w, _ = frame.shape
        
        # Get bounding box with same logic as recognition system
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        
        # Expand bounding box - SAME as recognition
        margin = 30
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        # Extract hand region
        hand_roi = frame[y_min:y_max, x_min:x_max]
        
        if hand_roi.size == 0:
            return None, "Empty ROI"
            
        # Convert to grayscale
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        
        # Create precise hand mask using MediaPipe landmarks
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Convert landmarks to ROI coordinates
        points = []
        for lm in hand_landmarks.landmark:
            x = int((lm.x * w - x_min))
            y = int((lm.y * h - y_min))
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                points.append([x, y])
        
        if len(points) < 10:  # Need enough points
            return None, "Insufficient landmarks"
        
        # Create convex hull mask - EXACT same method
        points = np.array(points)
        hull = cv2.convexHull(points)
        
        # Expand hull slightly for better coverage
        expanded_hull = []
        center = np.mean(hull, axis=0)[0]
        for point in hull:
            direction = point[0] - center
            expanded_point = point[0] + direction * 0.1  # 10% expansion
            expanded_hull.append(expanded_point)
        
        expanded_hull = np.array(expanded_hull, dtype=np.int32)
        cv2.fillPoly(mask, [expanded_hull], 255)
        
        # Quality control - check mask area
        mask_area = cv2.countNonZero(mask)
        if mask_area < self.min_hand_area or mask_area > self.max_hand_area:
            return None, f"Hand size issue: {mask_area} pixels"
        
        # Apply Otsu's thresholding on masked region - EXACT match
        if cv2.countNonZero(mask) > 50:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            clean_hand = cv2.bitwise_and(binary, binary, mask=mask)
        else:
            return None, "Insufficient mask pixels"
        
        # Resize to 28x28 - SAME method
        resized = cv2.resize(clean_hand, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Quality check - ensure hand is visible
        if np.sum(resized > 0) < 100:  # Minimum white pixels
            return None, "Hand too small in 28x28"
        
        return resized, "OK"
    
    def capture_variations(self, processed_image, letter):
        """Capture multiple variations of the same sample"""
        variations = []
        
        # Original
        variations.append(processed_image.copy())
        
        # Small rotations (-5° to +5°)
        for angle in [-3, -1, 1, 3]:
            M = cv2.getRotationMatrix2D((14, 14), angle, 1)
            rotated = cv2.warpAffine(processed_image, M, (28, 28))
            variations.append(rotated)
        
        # Small translations
        for tx, ty in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            translated = cv2.warpAffine(processed_image, M, (28, 28))
            variations.append(translated)
        
        # Slight brightness variations
        for brightness in [-15, -8, 8, 15]:
            bright = cv2.add(processed_image, brightness)
            bright = np.clip(bright, 0, 255).astype(np.uint8)
            variations.append(bright)
        
        return variations
    
    def run_enhanced_collection(self):
        """Enhanced collection with quality control and feedback"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return
        
        # Set camera properties for consistency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("OPTIMIZED ASL DATA COLLECTION")
        print("=" * 50)
        print("Enhancements:")
        print("✓ Exact preprocessing match")
        print("✓ Quality control checks") 
        print("✓ Multiple variations per sample")
        print("✓ Real-time feedback")
        print("-" * 50)
        print("Controls:")
        print("SPACE = Capture sample")
        print("N = Next letter")
        print("R = Reset current letter")
        print("Q = Quit")
        print("=" * 50)
        
        frame_count = 0
        good_samples = 0
        rejected_samples = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            
            # Process hands
            results = self.hands.process(rgb_frame)
            
            # Current status
            if self.current_letter_idx < len(self.letters):
                current_letter = self.letters[self.current_letter_idx]
                progress = f"{self.collected_count}/{self.samples_per_letter}"
            else:
                current_letter = "COMPLETE"
                progress = "DONE!"
            
            # Enhanced UI
            cv2.rectangle(frame, (10, 10), (500, 150), (0, 0, 0), -1)
            cv2.putText(frame, f"Letter: {current_letter}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f"Progress: {progress}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Good: {good_samples} | Rejected: {rejected_samples}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "SPACE=Capture | N=Next | R=Reset | Q=Quit", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Process hand detection with quality feedback
            processed_image = None
            status_message = "No hand detected"
            status_color = (0, 0, 255)  # Red
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                              self.mp_hands.HAND_CONNECTIONS)
                    
                    # Process with quality control
                    processed_image, status = self.process_hand_region_exact_match(frame, hand_landmarks)
                    
                    if processed_image is not None:
                        status_message = "✓ Ready to capture"
                        status_color = (0, 255, 0)  # Green
                        
                        # Show processed preview (larger)
                        display_processed = cv2.resize(processed_image, (120, 120), 
                                                     interpolation=cv2.INTER_NEAREST)
                        frame[160:280, 20:140] = cv2.cvtColor(display_processed, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(frame, (18, 158), (142, 282), (0, 255, 0), 2)
                        
                        # Show histogram for quality assessment
                        hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
                        hist_img = np.zeros((60, 256, 3), dtype=np.uint8)
                        cv2.normalize(hist, hist, 0, 60, cv2.NORM_MINMAX)
                        
                        for i in range(256):
                            cv2.line(hist_img, (i, 60), (i, 60-int(hist[i])), (255, 255, 255), 1)
                        
                        frame[300:360, 20:276] = hist_img
                        cv2.putText(frame, "Pixel Histogram", (20, 295), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                    else:
                        status_message = f"✗ {status}"
                        status_color = (0, 165, 255)  # Orange
            
            # Status display
            cv2.rectangle(frame, (10, 380), (500, 420), (0, 0, 0), -1)
            cv2.putText(frame, status_message, (20, 405), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.imshow('Optimized ASL Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') and processed_image is not None:
                # Capture with variations
                if self.current_letter_idx < len(self.letters):
                    variations = self.capture_variations(processed_image, current_letter)
                    for i, var_img in enumerate(variations):
                        self.capture_sample_with_metadata(var_img, current_letter, f"var_{i}")
                    good_samples += len(variations)
                    print(f"Captured {len(variations)} variations of {current_letter}")
            elif key == ord('n'):
                self.next_letter()
            elif key == ord('r'):
                # Reset current letter
                self.collected_count = 0
                print(f"Reset {current_letter} collection")
            
            # Auto quality feedback every 30 frames
            if frame_count % 30 == 0 and processed_image is None and results.multi_hand_landmarks:
                rejected_samples += 1
        
        cap.release()
        cv2.destroyAllWindows()
        self.save_optimized_dataset()
    
    def capture_sample_with_metadata(self, processed_image, letter, variation_id=""):
        """Capture sample with enhanced metadata"""
        label = self.current_letter_idx
        pixels = processed_image.flatten() / 255.0
        
        # Calculate quality metrics
        pixel_variance = np.var(pixels)
        white_pixel_ratio = np.sum(pixels > 0.5) / len(pixels)
        
        sample_data = {
            'label': label,
            'letter': letter,
            'pixels': pixels.tolist(),
            'variation': variation_id,
            'quality_variance': float(pixel_variance),
            'white_ratio': float(white_pixel_ratio),
            'timestamp': datetime.now().isoformat()
        }
        
        self.collected_data.append(sample_data)
        self.collected_count += 1
        
        # Auto-advance if enough samples
        if self.collected_count >= self.samples_per_letter:
            self.next_letter()
    
    def next_letter(self):
        """Move to next letter"""
        if self.collected_count > 0:
            self.current_letter_idx += 1
            self.collected_count = 0
            
            if self.current_letter_idx < len(self.letters):
                print(f"\\nMoving to: {self.letters[self.current_letter_idx]}")
            else:
                print("\\nCollection complete!")
    
    def save_optimized_dataset(self):
        """Save with quality analysis"""
        if not self.collected_data:
            print("No data collected!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Quality analysis
        total_samples = len(self.collected_data)
        avg_variance = np.mean([s['quality_variance'] for s in self.collected_data])
        avg_white_ratio = np.mean([s['white_ratio'] for s in self.collected_data])
        
        print(f"\\nDataset Quality Analysis:")
        print(f"Total samples: {total_samples}")
        print(f"Average variance: {avg_variance:.4f}")
        print(f"Average white ratio: {avg_white_ratio:.4f}")
        
        # Convert to training CSV
        import pandas as pd
        
        rows = []
        for sample in self.collected_data:
            row = [sample['label']] + sample['pixels']
            rows.append(row)
        
        columns = ['label'] + [f'pixel{i}' for i in range(784)]
        df = pd.DataFrame(rows, columns=columns)
        
        # Save to main data directory
        output_file = Path("data") / "optimized_sign_mnist_train.csv"
        Path("data").mkdir(exist_ok=True)
        
        df.to_csv(output_file, index=False)
        
        print(f"\\nDataset saved: {output_file}")
        print(f"Ready to train: python quick_train.py")
        
        # Update data path in training script
        self.update_training_script(output_file)
    
    def update_training_script(self, data_path):
        """Update training script to use optimized data"""
        script_path = Path("quick_train.py")
        if script_path.exists():
            content = script_path.read_text()
            updated = content.replace(
                "train_df = pd.read_csv('data/sign_mnist_train.csv')",
                f"train_df = pd.read_csv('{data_path}')"
            )
            script_path.write_text(updated)
            print(f"Updated {script_path} to use optimized data")

def main():
    collector = OptimizedASLCollector()
    collector.run_enhanced_collection()

if __name__ == "__main__":
    main()