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

K = np.float32([[8,0,0],[0,8,0],[0,0,1]])
K_inv = np.linalg.inv(K)

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
	F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC, 0.1, 0.99)

	# We select only inlier points
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]
	
	pt1 = np.array([[pts1[0][0]], [pts1[0][1]], [1]])
	pt2 = np.array([[pts2[0][0], pts2[0][1], 1]])
	print ("The fundamental matrix is")
	print (F)
	print ("Fundamental matrix error check: %f"%np.dot(np.dot(pt2,F),pt1))
	E = K.T.dot(F).dot(K)

	print ("The essential matrix is")
	print (E)
	return pts1,pts2,F,E


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

def degeneracyCheckPass(first_points, second_points, rot, trans):
	rot_inv = rot
	for first, second in zip(first_points, second_points):
		first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
		first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
		second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

		if first_3d_point[2] < 0 or second_3d_point[2] < 0:
			return False

	return True

for k in range(len(imagePaths)-1):
	imge1,imge2 = redim_image(imagePaths,k)	
	j=0
	while j < 3601:
		img1 = imge1[0:1200, j:j+1200]
		img2 = imge2[0:1200, j:j+1200]
		#cv.imwrite(f'{str(k)}{str(j)}.jpg',img1)
		img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
		img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
		print("côté : ",(j/1200)+1)
		
		pts1,pts2,matches,kp1,des1,kp2,des2 = sift_flann(img1,img2)
		pts1,pts2,F,E = ratio(matches,kp1,pts1,kp2,pts2)
		
		
		U, S, Vt = np.linalg.svd(E)
		W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
		first_inliers = []
		second_inliers = []
		for i in range(len(pts1)):
			first_inliers.append(K_inv.dot([pts1[i][0], pts1[i][1], 1.0]))
			second_inliers.append(K_inv.dot([pts2[i][0], pts2[i][1], 1.0]))

		# First choice: R = U * W * Vt, T = u_3
		R = U.dot(W).dot(Vt)
		T = U[:, 2]
		
		# Start degeneracy checks
		if not degeneracyCheckPass(first_inliers, second_inliers, R, T):
			# Second choice: R = U * W * Vt, T = -u_3
			T = - U[:, 2]
			if not degeneracyCheckPass(first_inliers, second_inliers, R, T):
				# Third choice: R = U * Wt * Vt, T = u_3
				R = U.dot(W.T).dot(Vt)
				T = U[:, 2]
				if not degeneracyCheckPass(first_inliers, second_inliers, R, T):
					# Fourth choice: R = U * Wt * Vt, T = -u_3
					T = - U[:, 2]

		print ("Translation matrix is")
		print (T)
		print ("Modulus is %f" % np.sqrt((T[0]*T[0] + T[1]*T[1] + T[2]*T[2])))
		print ("Rotation matrix is")
		print (R)
		
		# Decomposing rotation matrix
		pitch = np.arctan2(R[1][2], R[2][2]) * 180/3.1415
		yaw = np.arctan2(-R[2][0], np.sqrt(R[2][1]*R[2][1] + R[2][2]*R[2][2])) * 180/3.1415
		roll = np.arctan2(R[1][0],  R[0][0]) * 180/3.1415
		print ("Roll: %f, Pitch: %f, Yaw: %f" %(roll , pitch , yaw))

		# Find epilines corresponding to points in right image (second image) and
		# drawing its lines on left image
		lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
		lines1 = lines1.reshape(-1,3)
		img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
		
		# Find epilines corresponding to points in left image (first image) and
		# drawing its lines on right i", line 96, in mage
		lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
		lines2 = lines2.reshape(-1,3)
		img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
		
		plt.subplot(121),plt.imshow(img5)
		plt.subplot(122),plt.imshow(img3)
		plt.show()  
		
		

		j=j+1200

