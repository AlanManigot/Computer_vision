#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from imutils import paths
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import py360convert 
import json
import math

with open("sample.json","r") as f:
	data = json.load(f)
MIN_MATCH_COUNT = 250
imagePaths = data[0]["photos"]
k=0
print("nombre d'image : ",len(imagePaths))

K = np.mat([[8,0,0],[0,8,0],[0,0,1]])
print(K,k)
def redim_image(imagePaths,k):
	imge1 = cv.imread(data[0]["photos"][k]["localPath"])
	imge2 = cv.imread(data[0]["photos"][k+1]["localPath"])
	imge1 = py360convert.e2c(imge1, face_w=1200, mode='bilinear', cube_format='dice')
	imge2 = py360convert.e2c(imge2, face_w=1200, mode='bilinear', cube_format='dice')
	imge1 = cv.cvtColor(imge1, cv.COLOR_BGR2GRAY) 
	imge2 = cv.cvtColor(imge2, cv.COLOR_BGR2GRAY) 
	imge1 = imge1[1200:2400, 0:4800]
	imge2 = imge2[1200:2400, 0:4800]
	print("image1 = ",data[0]["photos"][k]["photoId"])
	print("image2 = ",data[0]["photos"][k+1]["photoId"])
	return imge1,imge2


def sift_flann(img1,img2):
	# Initiate SIFT detector
	sift = cv.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	# FLANN parameters
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	flann = cv.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	print(len(matches))
	pts1 = []
	pts2 = []
	return pts1,pts2,matches,kp1,des1,kp2,des2

def ratio(matches,kp1,pts1,kp2,pts2):
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.8*n.distance:
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	print(len(pts1),len(pts2))
	F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
	E, maskE = cv.findEssentialMat(pts1, pts2, focal=1, pp=(0,0), method=cv.RANSAC)
	_,R,t,maskR = cv.recoverPose(E,pts1,pts2)
	print("fondamental \n",np.mat(F))
	print("essential \n",np.mat(E))
	print("r : \n",np.mat(R),"\n t : \n",np.mat(t))
	pt1 = np.array([[pts1[0][0]], [pts1[0][1]], [1]])
	pt2 = np.array([[pts2[0][0], pts2[0][1], 1]])
	error=np.dot(np.dot(pt2,F),pt1)
	print("Fundamental matrix error  : \n",error)
	
	# We select only inlier points
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]
	
	# Computing Euler angles

	thetaX = np.arctan2(R[1][2], R[2][2])
	c2 = np.sqrt((R[0][0]*R[0][0] + R[0][1]*R[0][1]))

	thetaY = np.arctan2(-R[0][2], c2)

	s1 = np.sin(thetaX)
	c1 = np.cos(thetaX)

	thetaZ = np.arctan2((s1*R[2][0] - c1*R[1][0]), (c1*R[1][1] - s1*R[2][1]))

	print ("Pitch: %f, Yaw: %f, Roll: %f"%(thetaX*180/3.1415, thetaY*180/3.1415, thetaZ*180/3.1415))

	
	return pts1,pts2,F,R,t

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])



def drawlines(img1,img2,lines,pts1,pts2):
	''' img1 - image on which we draw the epilines for the points in img2
	lines - corresponding epilines '''
	r = img1.shape[0]
	c = img1.shape[1]  
	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
		img1 = cv.circle(img1,tuple(pt1),5,color,-1)
		img2 = cv.circle(img2,tuple(pt2),5,color,-1)
	return img1,img2  

def coord(F,R,t):
	line1 = F * (np.matrix ([15,16,1])).transpose ()
	line2 = F * (np.matrix ([341,532,1])).transpose ()
	a1,b1,c1 = line1.transpose ().tolist ()[0]
	a2,b2,c2 = line2.transpose ().tolist ()[0]
	x = - (c1 * b2 - c2 * b1) / (a1 * b2 - a2 * b1)
	y = - (a1 * c2 - a2 * c1) / (a1 * b2 - a2 * b1)	
	
	print("epipole x,y = ",x,", ",y)
	return

for k in range(len(imagePaths)-1):
	imge1,imge2 = redim_image(imagePaths,k)	
	j=0
	while j < 3601:
		img1 = imge1[0:1200, j:j+1200]
		img2 = imge2[0:1200, j:j+1200]
		cv.imwrite(f'{str(k)}{str(j)}.jpg',img1)
		img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
		img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
		print("côté : ",(j/1200)+1)
		
		pts1,pts2,matches,kp1,des1,kp2,des2 = sift_flann(img1,img2)
		pts1,pts2,F,R,t = ratio(matches,kp1,pts1,kp2,pts2)   
		Rv = rotationMatrixToEulerAngles(R)
		print("eul vector:",Rv)
		# Find epilines corresponding to points in right image (second image) and
		# drawing its lines on left image
		lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
		lines1 = lines1.reshape(-1,3)
		img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
		
		# Find epilines corresponding to points in left image (first image) and
		# drawing its lines on right image
		lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
		lines2 = lines2.reshape(-1,3)
		img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
			
		coord(F,R,t)

		points1 = [-line[2] / line[1] for line in lines1]
		points2 = [-line[2] / line[1] for line in lines2]
		distances = [abs (point1 - point2) for (point1, point2) in zip (points1, points2)]
		average1 = np.mean (distances)
		average2 = np.mean ([abs (distance - average1) for distance in distances])
		
		print("distance : ",average2)
		
		plt.subplot(121),plt.imshow(img5)
		plt.subplot(122),plt.imshow(img3)
		plt.show()  
		
		

		j=j+1200
