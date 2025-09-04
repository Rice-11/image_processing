
import cv2
import numpy as np
import os

class HandDetector:
    """
    Finds hands in a video frame using a pre-trained TensorFlow SSD model.
    """
    def __init__(self, model_path='models/frozen_inference_graph.pb', config_path='models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt', confidence_threshold=0.5):
        """
        Initializes the hand detector by loading the model.
        """
        self.confidence_threshold = confidence_threshold
        self.net = None

        # --- Explicitly check if model files exist ---
        model_exists = os.path.exists(model_path)
        config_exists = os.path.exists(config_path)

        if not model_exists or not config_exists:
            print("-" * 55)
            print("⚠️ ERROR: Hand Detector Model Files Not Found ⚠️")
            if not model_exists:
                print(f"- Missing model file: {model_path}")
            if not config_exists:
                print(f"- Missing config file: {config_path}")
            
            print("\nINSTRUCTIONS:")
            print("1. Download the necessary files:")
            print("   - Model:  https://drive.google.com/file/d/1s_n3iJc3t1U2a_3LNERso37oU33bJyr1/view")
            print("   - Config: https://raw.githubusercontent.com/cocodataset/d2ra/master/configs/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
            print("\n2. Place them in the 'models' directory in your project root.")
            print("   - The model file MUST be named 'frozen_inference_graph.pb'")
            print("   - The config file MUST be named 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'")
            print("-" * 55)
            return # Stop initialization

        # --- Try to load the network ---
        try:
            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            print("Hand detection model loaded successfully.")
        except cv2.error as e:
            print(f"Error loading hand detection model with OpenCV: {e}")
            print("The model files might be corrupted or incompatible.")

    def find_hands(self, frame, draw_box=True):
        """
        Finds hands in a frame and returns their bounding boxes.

        Args:
            frame: The video frame to search for hands.
            draw_box: Whether to draw the bounding box on the frame.

        Returns:
            A list of dictionaries, where each dictionary contains:
            'bbox': (x, y, w, h) of the detected hand.
            'confidence': The confidence score.
        """
        if self.net is None:
            return [], frame

        (h, w) = frame.shape[:2]
        
        # Preprocess the frame for the network
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/127.5, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()

        found_hands = []

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure the bounding box is within the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                box_w = endX - startX
                box_h = endY - startY

                if box_w > 0 and box_h > 0:
                    found_hands.append({
                        'bbox': (startX, startY, box_w, box_h),
                        'confidence': confidence
                    })

                    if draw_box:
                        # Draw the bounding box and confidence
                        label = f"Hand: {confidence:.2f}"
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return found_hands, frame
