#!/usr/bin/python3
# 2017.10.06 22:36:44 CST
# 2017.10.06 23:18:25 CST

"""
references: LearnOpenCV.com
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt

def objectDetection(imgname, imgname2):
 
  ## Create SIFT object
  sift = cv2.xfeatures2d.SIFT_create()

  ## Create flann matcher
  FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
  flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  #matcher = cv2.FlannBasedMatcher_create()
  matcher = cv2.FlannBasedMatcher(flann_params, {})

  ## Detect and compute
  img1 = cv2.imread(imgname)
  img1 = cv2.resize(img1, (300, 300), interpolation = cv2.INTER_CUBIC)
  img1_copy = img1.copy()
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  kpts1, descs1 = sift.detectAndCompute(gray1,None)

  ## As up
  img2 = cv2.imread(imgname2)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  kpts2, descs2 = sift.detectAndCompute(gray2,None)



  x_list = []
  y_list = []

  ## Ratio test
  matches = matcher.knnMatch(descs1, descs2, 2)
  matchesMask = [[0,0] for i in range(len(matches))]
  for i, (m1,m2) in enumerate(matches):
      if m1.distance < 0.7 * m2.distance:
          matchesMask[i] = [1,0]
          ## Notice: How to get the index
          pt1 = kpts1[m1.queryIdx].pt
          pt2 = kpts2[m1.trainIdx].pt
          # print(i, pt1,pt2 )

          x_list.append(pt1[0])
          # x_list.append(pt2[0])
          y_list.append(pt1[1])
          # y_list.append(pt2[1])

          if i % 5 ==0:
              ## Draw pairs in purple, to make sure the result is ok
              cv2.circle(img1, (int(pt1[0]),int(pt1[1])), 5, (255,0,255), -1)
              cv2.circle(img2, (int(pt2[0]),int(pt2[1])), 5, (255,0,255), -1)


  ## Draw match in blue, error in red
  draw_params = dict(matchColor = (255, 0,0),
                     singlePointColor = (0,0,255),
                     matchesMask = matchesMask,
                     flags = 0)


  x_list = [int(x) for x in x_list]
  y_list = [int(y) for y in y_list]


  # print(x_list)

  # if len(x_list) > 3:
  top_left = (min(x_list), max(y_list))
  bottom_right = (max(x_list), min(y_list))

  # img_boxed = img1_copy
  # img_boxed = cv2.rectangle(img_boxed,top_left,bottom_right,(0,255,0),3)
  # cv2.imshow("detected", img_boxed)

  # res = cv2.drawMatchesKnn(img1,kpts1,img2,kpts2,matches,None,**draw_params)
  # cv2.imshow("Result", res)


  cv2.waitKey()
  cv2.destroyAllWindows()
  return top_left,bottom_right
  return None,None
 

def detector():
  boxes = []

  for i in range(1,21):
    imgname = "dataset/img"+str(i)+".jpg"          # query image (large scene)
    imgname2 = "ref.jpg"   # train image (small object)

    pt1, pt2 = objectDetection(imgname, imgname2)
    boxes.append((pt1,pt2))

  return boxes

  

  


# detector()