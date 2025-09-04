import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)

print("Simple hand detection test - Press ESC to quit")
print("You should see green dots on your hand if detection is working")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands.process(rgb)
    
    # Draw results
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Show success message
        cv2.putText(frame, "HAND DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Hand Test', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()