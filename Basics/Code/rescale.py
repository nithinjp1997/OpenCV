import cv2 as cv


def rescaleFrame(frame, scale=0.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def changeRes(width, height):  # Only changes resolution for live video
    capture.set(3, width)
    capture.set(4, height)


# cap = cv.VideoCapture('Videos/Pexels Videos 2784.mp4')  # used to capture webcam vid or a vid path
# while True:
#     isTrue, frame = cap.read()  # used to read each frame of video
#     # Returns a boolean and the frame matrix for each frame
#     frame_resized = rescaleFrame(frame)
#     cv.imshow('Video', frame)  # displays the frame (same as in image)
#     cv.imshow('Video Resized', frame_resized)
#     if cv.waitKey(20) & 0xFF == ord('d'):  # Used to stop the frame read by pressing d
#         break
# cap.release()  # Releasing the video capture
# cv.destroyAllWindows()  # Closing all the windows

img = cv.imread('Photos/pic_large.jpg')
img_resized = rescaleFrame(img)
cv.imshow('Image', img)
cv.imshow('Image_resized', img_resized)
cv.waitKey(0)
