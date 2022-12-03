#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from imutils import paths
import py360convert
#parser
img1 = cv.imread('test.png',cv.IMREAD_GRAYSCALE)          
img2 = cv.imread('test2.png',cv.IMREAD_GRAYSCALE) 
#trasfo equi cubique mettre en png
#chercher les dimentions (ou redimentionner)
#diviser par 3 pour enlever y
j=0
while j < 1201:
    roi1 = img1[0:400, j:j+400]
    roi2 = img2[0:400, j:j+400]

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(roi1,None)
    kp2, des2 = sift.detectAndCompute(roi2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(roi1,kp1,roi2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()
    j=j+400



