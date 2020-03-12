# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:54:36 2020

@author: DELL
"""

import numpy as np 
import cv2

image = cv2.imread("E:\DCIM\Facebook\FB_IMG_1487519811046.jpg")

face_classifier = cv2.CascadeClassifier("F:\zipped\Computer-Vision-Tutorial-master\Haarcascades\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("F:\zipped\Computer-Vision-Tutorial-master\Haarcascadeshaarcascade_eye.xml")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = face_classifier.detectMultiScale(image)

if face is ():
  print("no face found")
for (x,y,w,h) in face:
  cv2.rectangle(image, (x,y),(x+w,y+h),(127,0,255),2,2)
  cv2.imshow("face", image)
  

  roi_gray = gray[y:y+h, x:x+w]
  roi_color = image[y:y+h, x:x+w]
  eye = eye_classifier.detectMultiScale(roi_gray)
  for (ex,ey,ew,eh) in eye:
    cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),(255,255,0),2)
    cv2.imshow("eye",image)
    
    cv2.waitKey()
cv2.destroyAllWindows()    

