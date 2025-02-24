import os
import cv2
import time
import numpy as np
from object_detection import load_yolo_model, detect_objects
from uart_comms import configure_uart, send_message, close_uart

def ensure_dialout_permissions():
    # Add the user to the dialout group if not already a member
    os.system("sudo usermod -aG dialout $USER")
    
    # Change the permissions of the UART port to ensure read and write access
    os.system("sudo chmod 666 /dev/ttyS0")

def main():
    ensure_dialout_permissions()
    counter = 0

    # Load both COCO and mask detection models
    coco_net, coco_classes, coco_output_layers = load_yolo_model('coco')
    mask_net, mask_classes, mask_output_layers = load_yolo_model('mask')

    #net, classes, output_layers = load_yolo_model()
    data = configure_uart()
    if data is None:
        print("Failed to open UART port. Exiting...")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video device")

    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Detection", 900, 640)

    # try:
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             print("Failed to grab frame")
    #             break

    #         if counter % 1 == 0:
    #             boxes, class_ids, confidences, indexes = detect_objects(net, output_layers, frame)

    #         for i in range(len(boxes)):
    #             if i in indexes:
    #                 x, y, w, h = boxes[i]
    #                 label = str(classes[class_ids[i]])
    #                 confidence = confidences[i]
    #                 color = (0, 255, 0)
    #                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    #                 cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #                 message = f"{label}: {x},{y},{w},{h}\n"
    #                 send_message(data, message)

    #         cv2.imshow("Object Detection", frame)

    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                if counter % 1 == 0:
                    coco_boxes, coco_class_ids, coco_confidences, coco_indices = detect_objects(coco_net, coco_output_layers, frame)
                    mask_boxes, mask_class_ids, mask_confidences, mask_indices = detect_objects(mask_net, mask_output_layers, frame)

                intruder_detected = False
                person_boxes = []

                for i in range(len(mask_boxes)):
                    if i in mask_indices:
                        x, y, w, h = mask_boxes[i]
                        label = str(mask_classes[mask_class_ids[i]])
                        confidence = mask_confidences[i]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        if label == "intruder":
                            intruder_detected = True
                            person_boxes.append(mask_boxes[i])

                if intruder_detected and person_boxes:
                    avg_x = int(np.mean([box[0] for box in person_boxes]))
                    avg_y = int(np.mean([box[1] for box in person_boxes]))
                    avg_w = int(np.mean([box[2] for box in person_boxes]))
                    avg_h = int(np.mean([box[3] for box in person_boxes]))
                    message = f"Intruder detected: intruder at ({avg_x}, {avg_y}, {avg_w}, {avg_h})"
                    send_message(data, message)

                    for i in range(len(coco_boxes)):
                        if i in coco_indices and coco_classes[coco_class_ids[i]] == "person":
                            x, y, w, h = coco_boxes[i]
                            avg_x = int(np.mean([box[0] for box in coco_boxes if coco_classes[coco_class_ids[i]] == "person"]))
                            avg_y = int(np.mean([box[1] for box in coco_boxes if coco_classes[coco_class_ids[i]] == "person"]))
                            avg_w = int(np.mean([box[2] for box in coco_boxes if coco_classes[coco_class_ids[i]] == "person"]))
                            avg_h = int(np.mean([box[3] for box in coco_boxes if coco_classes[coco_class_ids[i]] == "person"]))
                            send_message(data, f"Person detected: person at ({avg_x}, {avg_y}, {avg_w}, {avg_h})")
                            break

                if not intruder_detected:
                    for i in range(len(coco_boxes)):
                        if i in coco_indices:
                            x, y, w, h = coco_boxes[i]
                            label = str(coco_classes[coco_class_ids[i]])
                            confidence = coco_confidences[i]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            message = f"{label}: {x},{y},{w},{h}\n"
                            send_message(data, message)

                cv2.imshow("Object Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
            cap.release()
            cv2.destroyAllWindows()
            close_uart(data)

if __name__ == "__main__":
    main()
