import cv2
import math
import numpy as np
import serial
import time

# Setup serial communication with Arduino
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Adjust port if necessary
time.sleep(2)  # Wait for serial connection to stabilize

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_ssd.caffemodel")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

human_detected = False  # Track human detection status

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    height, width, _ = frame.shape
    middle_x = width // 2
    cv2.line(frame, (middle_x, 0), (middle_x, height), (0, 0, 255), 2)

    human_detected = False  # Reset detection flag

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] == "person":
                human_detected = True  # Human detected

                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                cv2.line(frame, (middle_x, height), (center_x, center_y), (0, 0, 255), 2)

                opposite = height - center_y
                adjacent = center_x - middle_x
                angle = math.degrees(math.atan2(opposite, adjacent)) - 90

                area = (x2 - x1) * (y2 - y1)
                area_threshold = 40000

                if area >= area_threshold:
                    command = "S"  # STOP
                elif angle >= 45:
                    command = "R"  # Turn RIGHT
                elif angle <= -45:
                    command = "L"  # Turn LEFT
                else:
                    command = "F"  # Move FORWARD

                print(f"Angle: {angle:.2f} | Area: {area} | Command: {command}")

                # Send command to Arduino
                arduino.write(command.encode())

    if not human_detected:
        print("No human detected. Spinning...")
        arduino.write("T".encode())  # Spin until human is found

    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
