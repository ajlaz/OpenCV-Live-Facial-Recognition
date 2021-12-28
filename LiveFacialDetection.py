#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:29:33 2021

@author: aden
"""
#This program uses live facial detection using opencv and the frontal face cascade.

import cv2
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap.set(3,640) #width
cap.set(4,480) #height
cap.set(10,100) #brightness
while True:
    
    success, img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w , y+h),(255,0,0),2)
    
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()