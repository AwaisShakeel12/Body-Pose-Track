import cv2
import mediapipe as mp

# Initialize mediapipe solutions
mp_Pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_Pose.Pose()
draw_spaces = mp_draw.DrawingSpec((0, 255, 0), thickness=5)

cap = cv2.VideoCapture(r"E:\computer vision\video\2795750-uhd_3840_2160_25fps.mp4")
# cap = cv2.VideoCapture(1)

while cap.isOpened():
    r, image = cap.read()

    if r:
        # Flip the image
        image = cv2.flip(image, 1)

        newimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = pose.process(newimg)

        if result.pose_landmarks:

            mp_draw.draw_landmarks(image, result.pose_landmarks, mp_Pose.POSE_CONNECTIONS, draw_spaces, draw_spaces)

            for id, lm in enumerate(result.pose_landmarks.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image,(cx,cy),5, (255,0,0), cv2.FILLED)

        image = cv2.resize(image, (600, 500))
        cv2.imshow('cap', image)

        if cv2.waitKey(25) & 0xFF == ord('p'):
            break
    else:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
