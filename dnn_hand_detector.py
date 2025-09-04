
import cv2
import numpy as np

# --- Model and Setup ---

# Define the paths to the pre-trained model files.
# This model is a TensorFlow-based Single Shot Detector (SSD) trained on the EgoHands dataset.
#
# WHERE TO DOWNLOAD THE MODEL FILES:
# 1. frozen_inference_graph.pb: This is the main model file.
#    You can download it from here: https://drive.google.com/file/d/1s_n3iJc3t1U2a_3LNERso37oU33bJyr1/view
# 2. ssd_mobilenet_v2_coco_2018_03_29.pbtxt: This is the text graph file that defines the model architecture.
#    A compatible version can be found in many public repositories. For example, from the official TensorFlow models repo:
#    https://github.com/opencv/opencv/blob/master/samples/dnn/models.yml (search for the filename)
#    Or a direct link: https://raw.githubusercontent.com/cocodataset/d2ra/master/configs/ssd_mobilenet_v2_coco_2018_03_29.pbtxt
#
# INSTRUCTIONS:
# 1. Download both files.
# 2. Place them in your 'models' directory.
# 3. Rename the .pbtxt file if you wish, but make sure the path in the script matches.

model_path = "models/frozen_inference_graph.pb"
config_path = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"

# Load the pre-trained model using OpenCV's DNN module.
try:
    net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
except cv2.error as e:
    print(f"Error loading model files: {e}")
    print(f"Please make sure the files '{model_path}' and '{config_path}' exist.")
    exit()

# --- Real-time Detection Logic ---

# Open the default webcam.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set a confidence threshold for detections.
confidence_threshold = 0.5

print("Starting webcam feed. Press 'q' to exit.")

while True:
    # Read a frame from the webcam.
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Get the frame dimensions.
    (h, w) = frame.shape[:2]

    # Preprocess the frame to create a blob for the neural network.
    # The model expects a 300x300 image.
    # Mean subtraction values (127.5, 127.5, 127.5) are typical for MobileNet models.
    # We also scale by 1/127.5 to normalize the pixel values to [-1, 1].
1    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/127.5, size=(300, 300), mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
    
    # Set the blob as input to the network.
    net.setInput(blob)

    # Perform a forward pass to get the detection results.
    detections = net.forward()

    # --- Processing Detections ---

    # Iterate through the detections.
    # The shape of the detections array is (1, 1, N, 7), where N is the number of detections.
    for i in range(detections.shape[2]):
        # Get the confidence score of the detection.
        confidence = detections[0, 0, i, 2]

        # If the confidence is above the threshold, it's a valid hand detection.
        if confidence > confidence_threshold:
            # The hand class ID in this model is typically 1.
            class_id = int(detections[0, 0, i, 1])

            # Although this model is for hands, we can add a check if needed.
            # if class_id == 1: # Or whatever the correct class ID for a hand is.
            
            # Calculate the coordinates for the bounding box.
            # The coordinates are relative to the image size, so we multiply by width and height.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and the confidence score on the frame.
            label = f"Hand: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Display and Exit ---

    # Display the final frame with bounding boxes.
    cv2.imshow("Hand Detection", frame)

    # Check if the 'q' key is pressed to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up: release the webcam and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
print("Application closed.")
