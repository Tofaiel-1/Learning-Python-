import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize the webcam and face mesh detector
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# A flag to prevent multiple clicks in a short time
last_click_time = time.time()

def draw_landmarks(frame, landmarks, indices, color):
    """Draw circles on specified landmarks."""
    for index in indices:
        x = int(landmarks[index].x * frame.shape[1])
        y = int(landmarks[index].y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, color, -1)

def main():
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        if landmark_points:
            landmarks = landmark_points[0].landmark

            # Draw eye landmarks
            draw_landmarks(frame, landmarks, range(474, 478), (0, 255, 0))

            # Control mouse movement
            eye_landmark = landmarks[475]
            screen_x = int(screen_w * eye_landmark.x)
            screen_y = int(screen_h * eye_landmark.y)
            pyautogui.moveTo(screen_x, screen_y)

            # Draw left eye landmarks
            draw_landmarks(frame, landmarks, [145, 159], (0, 255, 255))

            # Check for blink (simple approach)
            if abs(landmarks[145].y - landmarks[159].y) < 0.004:
                current_time = time.time()
                if current_time - last_click_time > 1:  # 1 second interval
                    pyautogui.click()
                    last_click_time = current_time

        cv2.imshow('Eye Controlled Mouse', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()