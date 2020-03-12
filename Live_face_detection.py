# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 01:09:55 2020

@author: DELL
"""

import numpy as np 
import cv2

face_classifier = cv2.CascadeClassifier("F:\zipped\Computer-Vision-Tutorial-master\Haarcascades\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("F:\zipped\Computer-Vision-Tutorial-master\Haarcascadeshaarcascade_eye.xml")



def detect(gray, frame):

  face = face_classifier.detectMultiScale(gray, 1.3, 5)

  for (x,y,w,h) in face:

      cv2.rectangle(frame, (x,y),(x+w,y+h),(127,0,255),2,2)
     
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = image[y:y+h, x:x+w]

      eye = eye_classifier.detectMultiScale(roi_gray, 1.1, 3)
      for (ex,ey,ew,eh) in eye:
          cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),(255,255,0),2)
  return frame
video_capture = cv2.VideoCapture(0)
while True:
  _,frame = video_capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  canvas = detect(gray, frame)
  cv2.imshow("canvas", canvas)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()