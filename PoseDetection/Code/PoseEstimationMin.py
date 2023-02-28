import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture('PoseDetection\Videos\2.mp4')
# Get the video's resolution
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Set the window size to match the video's aspect ratio
aspect_ratio = 4/3
window_height = 500
window_width = int(window_height * aspect_ratio)
cv.namedWindow('Image', cv.WINDOW_NORMAL)
cv.resizeWindow('Image', window_width, window_height)

while True:
    success, img = cap.read()
    if not success:
        # If the video has ended, reset the video to the beginning
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    cv.imshow("Image", img)
    cv.waitKey(0)

cap.release()
cv.destroyAllWindows()
