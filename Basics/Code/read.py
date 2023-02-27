import cv2 as cv
# Reading Images
# img = cv.imread('Basics\Photos/pic_large.jpg')  # imread() is the method to read images
#
# cv.imshow('Cat', img)  # imshow() displays the image .
# # 2 Parameters 1. window name, 2. image matrix
#
# cv.waitKey(0)

#Reading Videos

cap = cv.VideoCapture('Basics\Videos\Pexels Videos 2784.mp4') # used to capture webcam vid or a vid path
while True:
    isTrue, frame = cap.read() # used to read each frame of video
    # Returns a boolean and the frame matrix for each frame
    cv.imshow('Video', frame) # displays the frame (same as in image)
    if cv.waitKey(20) & 0xFF == ord('d'): # Used to stop the frame read by pressing d
        break
cap.release() # Releasing the video capture
cv.destroyAllWindows() # Closing all the windows
