import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]

def fingers_up(landmarks):
    fingers = []

    # Thumb
    if landmarks[tip_ids[0]][1] > landmarks[tip_ids[0]-1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for i in range(1,5):
        if landmarks[tip_ids[i]][2] < landmarks[tip_ids[i]-2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    letter = ""

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, c = img.shape
            landmarks = []

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                landmarks.append((id, cx, cy))

            fingers = fingers_up(landmarks)

            # Gesture rules
            if fingers == [0,1,0,0,0]:
                letter = "A"

            elif fingers == [0,1,1,0,0]:
                letter = "B"

            elif fingers == [0,1,1,1,0]:
                letter = "C"
            elif fingers == [0,1,1,1,1]:
                letter = "D"

            elif fingers == [1,1,1,1,1]:
                letter = "HELLO"
            elif fingers == [1,1,0,0,1]:
                letter = "yoyo Honey Singh"
            elif fingers == [0,0,1,1,1]:
                letter = "super"
            elif fingers == [0,0,0,0,0]:
                letter = "ooo"   

    cv2.putText(img, letter, (50,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2, (0,255,0), 3)

    cv2.imshow("Sign Language Detection", img)

    if cv2.waitKey(1) == 27:
        break