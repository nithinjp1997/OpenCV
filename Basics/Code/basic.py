import cv2 as cv

img = cv.imread('Basics\Photos\pic4.jpg')
cv.imshow('Image', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur (reduces noise)
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)  # The 2nd param is ksize(kernel size) it should be odd
# increasing the kernel size increases the blur. (I think this kernel is like the filter in a CNN.)
cv.imshow('Blur', blur)

# Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny', canny)

# Dilating the image
dilated = cv.dilate(canny, (3, 3), iterations=1)
cv.imshow('Dilated', dilated)

# Eroding 
eroded = cv.erode(dilated, (3, 3), iterations=1)
cv.imshow('Eroded',eroded)

# Resize and crop
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_LINEAR)
cv.imshow("resized", resized)

# Cropping 
cropped = img[50:200, 200:400]
cv.imshow("Crop",cropped)

cv.waitKey(0)
