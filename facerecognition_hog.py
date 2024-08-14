import cv2
import numpy as np
import os
import dlib

recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read('face_detection_hog/trainer/trainer.yml')
model = dlib.get_frontal_face_detector()

#id
id = 0

#names
names = ["None", "Rainier", "Jasmine", "Boris"]

#camera
#create and start video capture
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(3,640) #width
camera.set(4,480) #height
camera.set(cv2.CAP_PROP_FPS, 60.0)

#min window size to be recognised
minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)

#start of face detection and recognition
while True:
    success, img = camera.read()
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = model(img_grey)
    for(i, rect) in enumerate(faces):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        id, confidence = recogniser.predict(img_grey[y:y+h, x:x+w])
        
        #confidence < 100 = perfect match
        if(confidence<100):
            id = names[id]
            confidence = "  {0}%".format(round(100-confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100-confidence))
        
        #places text near the face
        cv2.putText(img, str(id), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    #show window
    cv2.imshow("Face Recognition via HOG", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#cleanup
camera.release()
cv2.destroyAllWindows