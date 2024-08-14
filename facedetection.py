import cv2
import numpy as np
import time

haarcascadeface = "./model/haarcascade_frontalface_default.xml"
haarcascadeeyes = "./model/haarcascade_eye_tree_eyeglasses.xml"
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(3, 640) #width
camera.set(4, 480) #height
camera.set(cv2.CAP_PROP_FPS, 60.0)

#assigning numeric face id for each person in the data sheet
#numeric face id will be used to grab the correct element from the name array for later
face_id = input('\n Enter user id end press <return> ==>  ')
print("\n Initialising face capture. Please look at the camera...")

count = 0
while True:
    success, img = camera.read()
    
    fps = camera.get(cv2.CAP_PROP_FPS)
    print(fps)
    
    #Loads models
    facemodel = cv2.CascadeClassifier(haarcascadeface)
    eyemodel = cv2.CascadeClassifier(haarcascadeeyes)
    
    #Converts input to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #face detection
    face = facemodel.detectMultiScale(img_gray, 1.1, 4)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

        count += 1
        #storing faces to dataset
        cv2.imwrite("./dataset/User." + str(face_id) + '.' + str(count) + ".jpg", img_gray[y:y+h, x:x+w])
        faceROI = img_gray[y:y+h, x:x+w]
        #eye detection
        eyes = eyemodel.detectMultiScale(faceROI)
        for(x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2+h2) * 0.25))
            cv2.circle(img, eye_center, radius, (0,255,255), 2)
        
    cv2.imshow('Webcam', img)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >=30: #Take 30 face samples and stop video
        break

#Cleanup
camera.release()
cv2.destroyAllWindows()
    
        