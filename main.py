import cv2
from hand_tracking import HandTracker
from game import Game

cap = cv2.VideoCapture(0)

tracker = HandTracker()
game = Game()

while True:

    success, img = cap.read()

    img, landmarks = tracker.detect_hands(img)

    finger_x = None
    finger_y = None
    erase = False

    if len(landmarks) != 0:

        id, x, y = landmarks[8]  # index finger tip
        finger_x = x
        finger_y = y

        mid_x, mid_y = landmarks[12][1], landmarks[12][2]

        # check if middle finger is raised (eraser mode)
        if mid_y < y:
            erase = True

        cv2.circle(img, (x, y), 10, (0, 0, 0), -1)

    img = game.update(img, finger_x, finger_y, erase)

    cv2.imshow("Air Draw", img)

    if cv2.waitKey(1) == 27:
        break