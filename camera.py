import cv2
import sys
import os
import os.path
import json
from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

app = ClarifaiApp()
model = app.models.get('boss')
model = model.train()
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
fullimg = cv2.imread("1.png")

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            flags=0|cv2.CASCADE_SCALE_IMAGE
            )
    #for(x,y,w,h) in faces:
    #        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.imshow('video',frame)
    if len(faces)>0:
        for(x,y,w,h) in faces:
            face = frame[y:y+h,x:x+w]
            cv2.imwrite('tmp.jpg',face)
            image = ClImage(file_obj=open('tmp.jpg','rb'))
            probability = model.predict([image])['outputs'][0]['data']['concepts'][0]['value']
            if probability>0.6:
                print('boss is coming')
        
                cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow("window",fullimg)
            else:
                print('not the boss')
    t = cv2.waitKey(1)

    if t==ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
