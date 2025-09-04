import cv2
import numpy as np

class OpenCVHandTracker:
    def __init__(self):
        # Initialize background subtractor for hand detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Skin color detection parameters (HSV)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Contour filtering parameters
        self.min_contour_area = 3000
        self.max_contour_area = 50000
        
    def detect_skin(self, frame):
        """Detect skin-colored regions in the frame"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Gaussian blur to smooth
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def find_hand_contours(self, mask):
        """Find hand contours from the mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        hand_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                hand_contours.append(contour)
        
        # Sort by area (largest first) and take up to 2 hands
        hand_contours.sort(key=cv2.contourArea, reverse=True)
        return hand_contours[:2]
    
    def draw_hand_landmarks(self, frame, contours):
        """Draw hand landmarks and contours"""
        for contour in contours:
            # Draw contour
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Find convex hull and defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            
            # Draw bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Find and draw centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 10, (255, 255, 0), -1)
                cv2.putText(frame, "Center", (cx-30, cy-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw convex hull points
            hull_points = cv2.convexHull(contour)
            for point in hull_points:
                cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)
            
            # Draw fingertips (convexity defects)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    # Draw lines
                    cv2.line(frame, start, end, [0, 255, 255], 2)
                    cv2.circle(frame, far, 8, [211, 84, 0], -1)
        
        return frame
    
    def process_frame(self, frame):
        """Process frame and detect hands"""
        # Create a copy for processing
        processed_frame = frame.copy()
        
        # Detect skin regions
        skin_mask = self.detect_skin(frame)
        
        # Find hand contours
        hand_contours = self.find_hand_contours(skin_mask)
        
        # Draw hand landmarks
        if hand_contours:
            processed_frame = self.draw_hand_landmarks(processed_frame, hand_contours)
            
            # Add hand count
            cv2.putText(processed_frame, f"Hands detected: {len(hand_contours)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show skin mask in corner
        mask_resized = cv2.resize(skin_mask, (150, 100))
        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        processed_frame[10:110, processed_frame.shape[1]-160:processed_frame.shape[1]-10] = mask_colored
        cv2.rectangle(processed_frame, (processed_frame.shape[1]-160, 10), 
                     (processed_frame.shape[1]-10, 110), (255, 255, 255), 2)
        cv2.putText(processed_frame, "Skin Mask", (processed_frame.shape[1]-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return processed_frame
    
    def run(self, camera_index=0):
        """Run the hand tracking application"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("OpenCV Hand Tracking started.")
        print("Instructions:")
        print("- Place your hand in front of the camera")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset background model")
        print("- Press 'c' to calibrate skin color")
        
        calibration_mode = False
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            if not calibration_mode:
                processed_frame = self.process_frame(frame)
            else:
                # Calibration mode - show original frame with instructions
                processed_frame = frame.copy()
                cv2.putText(processed_frame, "CALIBRATION MODE", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(processed_frame, "Place hand in center and press 'c' again", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw calibration area
                center_x, center_y = frame.shape[1]//2, frame.shape[0]//2
                cv2.rectangle(processed_frame, (center_x-50, center_y-50), 
                             (center_x+50, center_y+50), (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Hand Tracking', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset background subtractor
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
                print("Background model reset")
            elif key == ord('c'):
                if not calibration_mode:
                    calibration_mode = True
                    print("Calibration mode activated")
                else:
                    # Calibrate skin color from center region
                    center_x, center_y = frame.shape[1]//2, frame.shape[0]//2
                    roi = frame[center_y-50:center_y+50, center_x-50:center_x+50]
                    
                    if roi.size > 0:
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        
                        # Calculate new skin color range
                        mean_hsv = np.mean(hsv_roi.reshape(-1, 3), axis=0)
                        std_hsv = np.std(hsv_roi.reshape(-1, 3), axis=0)
                        
                        self.lower_skin = np.array([
                            max(0, mean_hsv[0] - 2*std_hsv[0]),
                            max(20, mean_hsv[1] - 2*std_hsv[1]),
                            max(70, mean_hsv[2] - 2*std_hsv[2])
                        ], dtype=np.uint8)
                        
                        self.upper_skin = np.array([
                            min(179, mean_hsv[0] + 2*std_hsv[0]),
                            255,
                            255
                        ], dtype=np.uint8)
                        
                        print(f"Skin color calibrated: {self.lower_skin} - {self.upper_skin}")
                    
                    calibration_mode = False
                    print("Calibration complete")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    tracker = OpenCVHandTracker()
    tracker.run()

if __name__ == "__main__":
    main()