import cv2
import numpy as np
import os

def load_yolo_model():
    weights_path = os.path.expanduser("~/darknet/yolov3-tiny.weights")
    config_path = os.path.expanduser("~/darknet/cfg/yolov3-tiny.cfg")
    names_path = os.path.expanduser("~/darknet/data/coco.names")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} not found")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{weights_path} not found")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"{names_path} not found")

    net = cv2.dnn.readNet(weights_path, config_path)

    classes = []
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers

def detect_objects(net, output_layers, frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
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
    return boxes, class_ids, confidences, indexes