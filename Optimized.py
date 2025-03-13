import cv2
import math
import numpy as np

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_ssd.caffemodel")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Use OpenCV optimizations
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Run on CPU (best for Raspberry Pi)

# Class labels (full list to avoid indexing errors)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Initialize video capture (use default webcam)
cap = cv2.VideoCapture(0)

# Set webcam resolution (reduces processing load)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Adjust brightness & contrast (optimized for webcam)
    frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Get frame dimensions
    height, width, _ = frame.shape
    middle_x = width // 2

    # Draw center line
    cv2.line(frame, (middle_x, 0), (middle_x, height), (0, 0, 255), 2)

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:  # Lower threshold for better Raspberry Pi performance
            idx = int(detections[0, 0, i, 1])

            if 0 <= idx < len(CLASSES) and CLASSES[idx] == "person":
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")

                # Draw bounding box & center point
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                cv2.line(frame, (middle_x, height), (center_x, center_y), (0, 0, 255), 2)

                # Calculate angle
                opposite = height - center_y
                adjacent = center_x - middle_x
                angle = math.degrees(math.atan2(opposite, adjacent))
                corrected_angle = angle - 90

                # Calculate bounding box area
                area = (x2 - x1) * (y2 - y1)
                area_threshold = 40000  # Adjust for Raspberry Pi

                # Determine movement
                if area >= area_threshold:
                    command = "STOP"
                elif corrected_angle >= 45:
                    command = "LEFT"
                elif corrected_angle <= -45:
                    command = "RIGHT"
                else:
                    command = "FORWARD"

                print(f"Angle: {corrected_angle:.2f} | Area: {area} | Command: {command}")

    # Show output
    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
