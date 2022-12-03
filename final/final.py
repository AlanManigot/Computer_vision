#!/usr/bin/env python
# coding: utf-8

#package
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from imutils import paths

#variables
i=0
MIN_MATCH_COUNT = 4
imageNames = sorted(list(paths.list_images("images")))

def size(image):
	for i in range(len(imageNames)):
		imageNames[i] = imageNames[i][imageNames[i].find('/')+1:-4]

