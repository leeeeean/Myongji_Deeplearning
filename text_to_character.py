import numpy as np
import argparse
import time
import cv2

#read image
img = cv2.imread('sign.jpg')

#grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#binarize 
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

#find contours
im2,ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = img[y:y+h, x:x+w]

    # show ROI
    #cv2.imwrite('roi_imgs.png', roi)
    cv2.imshow('charachter'+str(i), roi)
    cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)

cv2.imshow('marked areas',img)
cv2.waitKey(0)

