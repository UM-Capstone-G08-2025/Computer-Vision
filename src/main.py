import os
import cv2
import time
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
                # Detect objects using COCO model
                coco_boxes, coco_class_ids, coco_confidences, coco_indexes = detect_objects(coco_net, coco_output_layers, frame)
                # Detect objects using mask detection model
                mask_boxes, mask_class_ids, mask_confidences, mask_indexes = detect_objects(mask_net, mask_output_layers, frame)

            intruder_detected = False

            # Draw mask detections and check for intruders
            for i in range(len(mask_boxes)):
                if i in mask_indexes:
                    x, y, w, h = mask_boxes[i]
                    label = str(mask_classes[mask_class_ids[i]])
                    confidence = mask_confidences[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    if label == "intruder":
                        intruder_detected = True
                        send_message(data, f"Intruder detected: {label} at ({x}, {y}, {w}, {h})")

            # Draw COCO detections and send data if no intruder detected
            if not intruder_detected:
                for i in range(len(coco_boxes)):
                    if i in coco_indexes:
                        x, y, w, h = coco_boxes[i]
                        label = str(coco_classes[coco_class_ids[i]])
                        confidence = coco_confidences[i]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        send_message(data, f"Object detected: {label} at ({x}, {y}, {w}, {h})")

            cv2.imshow("Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            counter += 1
    except KeyboardInterrupt:
        print("\nTransmission interrupted by user.")
    finally:
        close_uart(data)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()