import cv2
import numpy as np
import dlib

#model
hogModel = dlib.get_frontal_face_detector()

#assigning numeric face id for each person in the data sheet
#numeric face id will be used to grab the correct element from the name array for later
face_id = input('\n Enter user id end press <return> ==>  ')
print("\n Initialising face capture. Please look at the camera...")

count = 0

#setting camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(3, 640) #width
camera.set(4, 480) #height
camera.set(cv2.CAP_PROP_FPS, 60.0)

while True:
    success, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = hogModel(img_gray)
    
    for(i, rect) in enumerate(face):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1
        #storing faces to dataset
        cv2.imwrite("face_detection_hog/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", img_gray[y:y+h, x:x+w])
        
    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count>=30:
        break

#Cleanup
camera.release()
cv2.destroyAllWindows()