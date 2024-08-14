import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.png')   # you can read in images with opencv
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv_color1 = np.asarray([36, 50, 70])  
hsv_color2 = np.asarray([89, 255, 255]) 
mask = cv2.inRange(img_hsv, hsv_color1, hsv_color2)

_, thresh=cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.dilate(thresh, (kernel)) 

contours, _=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
	x,y,w,h = cv2.boundingRect(c)
	center = (x,y)
	print(center)

cv2.imshow("mask", thresh)
cv2.waitKey(0)

