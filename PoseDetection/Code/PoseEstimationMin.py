import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture("PoseDetection\Videos\Vid7.mp4")

# h = int(cap.get(cv.CAP_PROP_FOURCC))
# codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
# print(codec)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
# Get the video's resolution
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Set the window size to match the video's aspect ratio
aspect_ratio = width / height
window_height = 600
window_width = int(window_height * aspect_ratio)
cv.namedWindow('Image', cv.WINDOW_NORMAL)
cv.resizeWindow('Image', window_width, window_height)
pTime = 0

while (cap.isOpened()):
    success, img = cap.read()
    if not success:
        # If the video has ended, reset the video to the beginning
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED) 

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv.imshow("Image", img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
