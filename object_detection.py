import cv2
import numpy as np
import os
import time

# Define paths to the weights, config, and names files
weights_path = os.path.expanduser("~/darknet/yolov3-tiny.weights")
config_path = os.path.expanduser("~/darknet/cfg/yolov3-tiny.cfg")
names_path = os.path.expanduser("~/darknet/data/coco.names")

# Check if files exist
if not os.path.exists(config_path):
    raise FileNotFoundError(f"{config_path} not found")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"{weights_path} not found")
if not os.path.exists(names_path):
    raise FileNotFoundError(f"{names_path} not found")

print("Loading YOLO model...")
# Load YOLO model and weights
net = cv2.dnn.readNet(weights_path, config_path)

# Load COCO classes
classes = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

print("Starting video capture...")

# Initialize variables for FPS calculation
frame_id = 0
start_time = time.time()

# Create a named window and resize it
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", 900, 640)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_id += 1
    height, width, _ = frame.shape

    # YOLO input preprocessing
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"Detected {label} at [{x}, {y}, {w}, {h}] with confidence {confidences[i]}")

    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate and display average confidence
    if confidences:
        avg_confidence = sum(confidences) / len(confidences) * 100
        cv2.putText(frame, f"Avg Confidence: {avg_confidence:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
print("Video capture ended.")