#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
import argparse
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	cv2.imshow("edged", edged)
	cv2.waitKey(0)
	return cv2.minAreaRect(c)
	


# In[ ]:
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="image.jpg",
	help="path to input image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])




# In[ ]:




