# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 00:40:16 2020

@author: saradhi
"""

from imageai.Detection import ObjectDetection
import os
import cv2
#from google.colab.patches import cv2_imshow
cap=cv2.VideoCapture("sample1.mp4")
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join("resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
while True:
  ret,img=cap.read()
  cv2.imwrite("image1.jpg",img) 
  detections = detector.detectObjectsFromImage(input_image="image1.jpg", output_image_path="imageout1.jpg")
  out=cv2.imread("imageout1.jpg")
  cv2.imshow("window",out)
  #cv2_imshow(out)
  if cv2.waitKey(1)==ord('q'):
    break
  for eachObject in detections:
     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
cap.release()
cv2.destroyAllWindows() 