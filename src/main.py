import cv2
import numpy as np
import os
from datetime import datetime

# def load_yolo_model():
#     weights_path = os.path.expanduser("~/Desktop/src/yolo-mask-detection/models/mask-yolov3-tiny-prn.weights")
#     config_path = os.path.expanduser("~/Desktop/src/yolo-mask-detection/models/mask-yolov3-tiny-prn.cfg")
#     names_path = os.path.expanduser("~/Desktop/src/yolo-mask-detection/models/mask.names")

def load_yolo_model(model_type='coco'):
    if model_type == 'coco':
        weights_path = os.path.expanduser("~/darknet/yolov3-tiny.weights")
        config_path = os.path.expanduser("~/darknet/cfg/yolov3-tiny.cfg")
        names_path = os.path.expanduser("~/darknet/data/coco.names")
    elif model_type == 'mask':
        weights_path = os.path.expanduser("~/Desktop/src/yolo-mask-detection/models/mask-yolov3-tiny-prn.weights")
        config_path = os.path.expanduser("~/Desktop/src/yolo-mask-detection/models/mask-yolov3-tiny-prn.cfg")
        names_path = os.path.expanduser("~/Desktop/src/yolo-mask-detection/models/mask.names")
    else:
        raise ValueError("Invalid model type. Choose 'coco' or 'mask'.")

    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} not found")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{weights_path} not found")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"{names_path} not found")

    #print(cv2.cuda.getCudaEnabledDeviceCount())
    net = cv2.dnn.readNet(weights_path, config_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    #print(cv2.getBuildInformation())

    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()

        if count > 0:
            print("Using cuda")
        else:
            print("not using cuda")

    except:
        print("Not using cuda")

    classes = []
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers

def detect_objects(net, output_layers, frame):
    # print(f"0: {datetime.now()}")
    height, width, _ = frame.shape
    # print(f"1: {datetime.now()}")
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # print(f"2: {datetime.now()}")
    net.setInput(blob)
    # print(f"3: {datetime.now()}")
    # Set OpenCL as the target for inference
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #inet.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    outs = net.forward(output_layers)
    # print(f"4: {datetime.now()}")
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

    # print(f"5: {datetime.now()}")
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(f"6: {datetime.now()}")

    return boxes, class_ids, confidences, indexes
