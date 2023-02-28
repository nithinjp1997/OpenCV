import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('HandDetection\HandVideo\production ID_3796261.mp4')
# Get the video's resolution
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Set the window size to match the video's aspect ratio
aspect_ratio = 4/3
window_height = 500
window_width = int(window_height * aspect_ratio)
cv.namedWindow('Image', cv.WINDOW_NORMAL)
cv.resizeWindow('Image', window_width, window_height)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    # print(results.multi_hand_landmarks)

    if not success:
        # If the video has ended, reset the video to the beginning
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 4:
                    cv.circle(img, (cx, cy), 25, (255, 0, 255), cv.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    cv.imshow('Image', img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
