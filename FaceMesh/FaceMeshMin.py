import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('FaceMesh\Videos\Vid (2).mp4')
# Get the video's resolution
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Set the window size to match the video's aspect ratio
aspect_ratio = width/height
window_height = 600
window_width = int(window_height * aspect_ratio)
cv.namedWindow('Image', cv.WINDOW_NORMAL)
cv.resizeWindow('Image', window_width, window_height)

pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(color = (255, 0, 255), thickness = 1, circle_radius = 1)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        # If the video has ended, reset the video to the beginning
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, 
                                  drawSpec, drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)

    cTime = time.time()
    fps = 1/(cTime- pTime)
    pTime = cTime
    cv.putText(img, f"FPS: {int(fps)}",(20, 70), cv.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0), 3)
    cv.imshow("Image", img)
    if cv.waitKey(1) == ord('q'):
        break