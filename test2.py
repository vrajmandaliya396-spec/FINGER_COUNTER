import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

        # 👉 Finger Logic (after full list ready)
        if len(lmList) >= 21:

            fingers = []

            # Index
            if lmList[8][2] < lmList[6][2] < lmList[5][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Middle
            if lmList[12][2] < lmList[10][2] < lmList[9][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Ring
            if lmList[16][2] < lmList[14][2] < lmList[13][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Pinky
            if lmList[20][2] < lmList[18][2] < lmList[17][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 👉 Thumb (different logic - x axis)
            if lmList[4][1] > lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            totalFingers = fingers.count(1)

            # 👉 Show count
            cv2.putText(img, str(totalFingers), (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) == ord('q'):
        break