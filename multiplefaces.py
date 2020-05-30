# -*- coding: utf-8 -*-
"""
Created on Sat May 23 13:16:31 2020

@author: phane
"""

import cv2
import numpy as np
from fastai import *
from fastai.vision import *
from PIL import Image
import numpy as np

prediction = ""


path = "/home/phaneendra/Downloads/withwithout-mask/maskdata/maskdata/"

tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path,ds_tfms=tfms,size=224,bs=16)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def testlearner(filename):

    print("Checking for mask")
    
    data = ImageDataBunch.from_folder(path,ds_tfms=tfms,size=224)
    
    learn = cnn_learner(data,models.resnet34).load('/home/phaneendra/Documents/python/fastai/mask-final')

    img = open_image(filename)

    prediction,pred_idx,output = learn.predict(img)

    print(prediction,pred_idx,output)

    return prediction


def ageandgender(image):
    
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    blob = cv2.dnn.blobFromImage(image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
    print("Age Range: " + age)

    return gender,age



def CaptureWebcam():

    capture = cv2.VideoCapture(0)

    ret_val,image = capture.read()

    tracker = cv2.TrackerMedianFlow_create()

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier("/home/phaneendra/Documents/python/haar-cascade-files-master/haarcascade_frontalface_alt2.xml")

    faces = face_cascade.detectMultiScale(gray,1.1,5)

    count = 0


    if(len(faces)>0):
        
        print("faces detected: " + str(len(faces)))
        
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
            
            roi = (x,y,w,h)
            
            
        
        roi = tuple(roi)
        
        
        
        ret = tracker.init(image,roi)
        
    while True:
        
        ret,frame = capture.read()
        
        success,roi = tracker.update(frame)
        
        print(roi)
        
        (x,y,w,h) = tuple(map(int,roi))
        
        if success:
            img = Image.fromarray(frame[y:y+h,x:x+w])
            filename =r'/home/phaneendra/Documents/python/faceimages/sample'+str(count)+r'.jpg' 
            img.save(filename)
            age,gender = ageandgender(frame[y:y+h,x:x+w])
            prediction = testlearner(filename)
            print(prediction)
            print(roi)
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            if (str(prediction) == "without_mask"):
                cv2.putText(frame,str(prediction),(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                cv2.putText(frame,age,(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                cv2.putText(frame,gender,(300,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            else:
                cv2.putText(frame,str(prediction),(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                cv2.putText(frame,age,(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                cv2.putText(frame,gender,(300,400),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            
            cv2.putText(frame,"Unable to track",(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            
        cv2.imshow("Tracker",frame)

        count = count + 1
        
        k = cv2.waitKey(1) & 0xff   
        
        if(k==27):
            break
        
    capture.release()

    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    CaptureWebcam()    

