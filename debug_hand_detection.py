import cv2
import mediapipe as mp

class HandDetectionDebug:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("Hand detection started. You should see:")
        print("- Green landmarks on your hands")
        print("- Connecting lines between landmarks")
        print("- Press 'q' to quit")
        
        frame_count = 0
        hands_detected_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            frame_count += 1
            
            # Flip the frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB (MediaPipe requirement)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            # Draw debug info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Check if hands are detected
            if results.multi_hand_landmarks:
                hands_detected_count += 1
                cv2.putText(frame, f"HANDS DETECTED: {len(results.multi_hand_landmarks)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks with connections
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style()
                    )
                    
                    # Draw bounding box
                    h, w, c = frame.shape
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
                    
                    cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), (255, 0, 0), 2)
                    cv2.putText(frame, "HAND", (x_min, y_min-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "NO HANDS DETECTED", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show detection success rate
            if frame_count > 0:
                success_rate = (hands_detected_count / frame_count) * 100
                cv2.putText(frame, f"Detection Rate: {success_rate:.1f}%", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Hand Detection Debug', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nSession Summary:")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with hands detected: {hands_detected_count}")
        print(f"Overall detection rate: {(hands_detected_count/frame_count)*100:.1f}%" if frame_count > 0 else "No frames processed")

if __name__ == "__main__":
    detector = HandDetectionDebug()
    detector.run()