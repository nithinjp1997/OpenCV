import cv2 as cv
import mediapipe
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0

cap = cv.VideoCapture('HandDetection\Videos\Pexels Videos 2784.mp4')
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# Set the window size to match the video's aspect ratio
aspect_ratio = 4/3
window_height = 500
window_width = int(window_height * aspect_ratio)
cv.namedWindow('Image', cv.WINDOW_NORMAL)
cv.resizeWindow('Image', window_width, window_height)

detector = htm.handDetector()
while True:
    success, img = cap.read()
    
    if not success:
        # If the video has ended, reset the video to the beginning
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    
    if len(lmList) != 0 :
        print(lmList[4])
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