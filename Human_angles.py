import cv2
import math
import numpy as np
import serial
import time

# Setup serial communication with Arduino
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)  # Wait for serial connection to stabilize
except serial.SerialException:
    print("Error: Could not connect to Arduino. Check the port and connection.")
    arduino = None  # Handle case where Arduino is not connected

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(
    "/home/vishwaraspi/mobilenet_ssd/deploy.prototxt",
    "/home/vishwaraspi/mobilenet_ssd/mobilenet_iter_73000.caffemodel"
)

# Set backend to OpenCV CPU (can switch to CUDA if needed)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_command = None  # Store last command to avoid spamming

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Warning: Failed to capture frame. Retrying...")
        continue  # Skip iteration if frame isn't captured properly

    # Adjust brightness/contrast dynamically (helps in low light)
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

    # Convert frame to blob for DNN processing
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    height, width, _ = frame.shape
    middle_x = width // 2

    cv2.line(frame, (middle_x, 0), (middle_x, height), (0, 0, 255), 2)

    human_detected = False
    new_command = "T"  # Default to "T" (Spin) if no person is found

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:  # Increased confidence threshold for better accuracy
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] == "person":
                human_detected = True

                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")

                # Draw bounding box and center point
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                cv2.line(frame, (middle_x, height), (center_x, center_y), (0, 0, 255), 2)

                # Calculate angle and distance (size of bounding box)
                opposite = height - center_y
                adjacent = center_x - middle_x
                angle = math.degrees(math.atan2(opposite, adjacent)) - 90

                area = (x2 - x1) * (y2 - y1)
                area_threshold = 80000  # Adjusted for larger frame size

                # Decide movement
                if area >= area_threshold:
                    new_command = "S"  # STOP (Person is too close)
                elif angle >= 45:  
                    new_command = "R"  # Turn RIGHT (45 degrees)
                elif angle <= -40:  
                    new_command = "L"  # Turn LEFT (45 degrees)
                else:
                    new_command = "F"  # Move FORWARD

                print(f"Angle: {angle:.2f} | Area: {area} | Command: {new_command}")

    if not human_detected:
        print("No human detected. Spinning...")
        new_command = "T"  # Spin if no human is found

    # Send command **only if different from last command**
    if new_command != last_command and arduino:
        try:
            arduino.write(new_command.encode())
            print(f"Sent command: {new_command}")

            # If turning, wait for **precise time** to turn 45°
            if new_command in ["L", "R"]:
                time.sleep(2)  # Adjust based on actual turning speed

        except serial.SerialException:
            print("Error: Serial connection lost. Command not sent.")

    last_command = new_command  # Store last command

    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if arduino:
    arduino.close()
