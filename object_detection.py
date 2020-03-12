# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 01:19:48 2020

@author: DELL
"""

import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
body_classifier = cv2.CascadeClassifier("F:\zipped\Computer-Vision-Tutorial-master\Haarcascades\haarcascade_fullbody.xml")
walk = cv2.VideoCapture("F:\zipped\Computer-Vision-Tutorial-master\image_examples\cars.avi")
while walk.isOpened():
    
    ret,frame = walk.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    body = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    
    for (x,y,w,h) in body:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255), 2)
      cv2.imshow("frame", frame)
    
      cv2.waitKey()
walk.release()
cv2.destroyAllWindows()  