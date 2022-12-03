#!/usr/bin/env python
# coding: utf-8

# In[32]:


#!/usr/bin/env python
# coding: utf-8

from imutils import paths
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import py360convert 

MIN_MATCH_COUNT = 8
imagePaths = sorted(list(paths.list_images("images")))
i=0
print(len(imagePaths))

for i in range(len(imagePaths)-1):
    imge1 = cv.imread(imagePaths[i])
    imge2 = cv.imread(imagePaths[i+1])
    imge1 = py360convert.e2c(imge1, face_w=1200, mode='bilinear', cube_format='dice')
    imge2 = py360convert.e2c(imge2, face_w=1200, mode='bilinear', cube_format='dice')
    imge1 = imge1[1200:3600, 0:4800]
    imge2 = imge2[1200:3600, 0:4800]
    j=0
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    while j < 3601:
        img1 = imge1[0:1200, j:j+1200]
        img2 = imge2[0:1200, j:j+1200]
        print(imagePaths[i])
        print(imagePaths[i+1])

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h = img1.shape[0]
            w = img1.shape[1]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)

            img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
            plt.imshow(img3, 'gray'),plt.show()
            cv.waitKey(0)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            print(j)
            matchesMask = None
    
        j=j+1200




# In[ ]:





# In[ ]:




