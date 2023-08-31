import cv2
import os

import numpy as np


def load_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara")
    else:
        while True:
            ret, frame = cap.read()
            processed_frame = recognition_eyeglass(frame)

            if processed_frame is not None:
                display_frame = combine_frames(frame, processed_frame)
                cv2.imshow("Camera", display_frame)
            else:
                cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


def recognition_eyeglass(frame):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_directory, 'CascadeClassifier/haarcascade_eye.xml')
    classifier = cv2.CascadeClassifier(xml_path)

    scaleFactor = 1.2
    minNeighbors = 4
    minSize = (30, 30)

    boxes = classifier.detectMultiScale(frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

    for box in boxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)

        eye_roi = frame[y:y2, x:x2]
        gray_eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        blurred_eye_roi = cv2.GaussianBlur(gray_eye_roi, (7, 7), 0)

        circles = cv2.HoughCircles(
            blurred_eye_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=30)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(eye_roi, center, radius, (255, 0, 0), 2)

    if len(boxes) > 0:
        x, y, width, height = boxes[0]
        eye_roi = frame[y:y + height, x:x + width]
        return eye_roi
    else:
        return None


def combine_frames(original_frame, processed_frame):
    # Resize the processed frame to a smaller size for displaying in the corner
    processed_frame_resized = cv2.resize(processed_frame, (100, 75))

    # Extract the dimensions of the original frame
    height, width, _ = original_frame.shape

    # Calculate the position for placing the processed frame in the corner
    pos_x = 10
    pos_y = 10

    # Copy the original frame for overlaying
    combined_frame = original_frame.copy()

    # Place the processed frame in the specified position
    combined_frame[pos_y:pos_y + 75, pos_x:pos_x + 100] = processed_frame_resized

    return combined_frame


if __name__ == '__main__':
    load_camera()
