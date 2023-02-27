import cv2 as cv
import numpy as np

img = cv.imread('Basics/Photos/pic1.jpg')
cv.imshow("Image", img)

blank = np.zeros(img.shape[:2], dtype='uint8')  # The shape of the mask must be the same as the image

circle = cv.circle(blank.copy(), (img.shape[1]//2,img.shape[0]//2), 200, 255, -1)
rectangle = cv.rectangle(blank.copy(), (100,0), (200, 200), 255, -1)
# cv.imshow("Mask", mask)
# cv.imshow("Rectangle", rectangle)
wierd_shape = cv.bitwise_and(rectangle, circle)
# cv.imshow("wierd_shape", wierd_shape)

masked = cv.bitwise_and(img, img, mask = wierd_shape)
cv.imshow("Masked", masked)

cv.waitKey(0)