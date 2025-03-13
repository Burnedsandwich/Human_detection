import cv2
import math
import numpy as np

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_ssd.caffemodel")

# List of class labels (MobileNet SSD supports 20 classes)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == False :
        break

    def adjust_brightness_contrast(image, alpha=2.0, beta=50):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


    frame = adjust_brightness_contrast(frame)

    # Convert frame to blob (prepares it for DNN model)
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Get frame dimensions
    height, width,_ = frame.shape

    # Compute the middle of the frame
    middle_x = width // 2

    # Draw a vertical line in the center
    cv2.line(frame, (middle_x, 0), (middle_x, height), (0, 0, 255), 3)

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Only consider detections > 50% confidence
            idx = int(detections[0, 0, i, 1])  # Class index
            if CLASSES[idx] == "person":  # Detect only humans
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x1, y1, x2, y2) = box.astype("int")

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.line(frame, (middle_x, height), (center_x, center_y), (0, 0, 255), 3)
                label = f"Person: {confidence * 100:.2f}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                opposite = height - center_y
                adjacent = center_x - middle_x
                angle = math.degrees(math.atan2(opposite, adjacent))
                cort_angle =(angle - 90)
                cv2.putText(frame, f"Angle: {cort_angle:.2f} deg", (center_x, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                # Calculate bounding box area
                area = (x2 - x1) * (y2 - y1)

                # Define area threshold (Adjust based on your setup)
                area_threshold = 50000  # Increase if needed

                # Determine movement based on angle & area
                if area >= area_threshold:
                    command = "STOP"
                elif cort_angle >= 45:
                    command = "LEFT"
                elif cort_angle <= -45:
                    command = "RIGHT"
                else:
                    command = "FORWARD"

                print(f"Angle: {cort_angle:.2f} | Area: {area} | Command: {command}")

        #if confidence > 0.9:
            #cv2.imwrite("captured_image.jpg", frame)
            #print("human detected FS")
    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
