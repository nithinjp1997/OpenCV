import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Basics\Photos\pic1.jpg')
# cv.imshow("Image", img)

rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("RGB", rgb)

plt.imshow(rgb)
plt.show()

# # BGR to Grayscale
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Gray", gray)

# # BGR to HSV
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow("HSV", hsv)

# # BGR to L*a*b 
# lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('Lab', lab)

# cv.waitKey(0)