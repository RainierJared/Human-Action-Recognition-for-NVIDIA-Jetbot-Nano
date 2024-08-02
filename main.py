import cv2
import mediapipe as mp
import numpy as np
import time
import pickle

fileName = './test_video/sitting-test.mp4'
cap = cv2.VideoCapture(fileName)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

labelsDict = {'walking': 'walk', 'sitting': 'sitting'}

model_dict = pickle.load(open('./model/model.p', 'rb'))
model = model_dict['model']

pTime = 0

while True:
    success, img = cap.read()
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    temp = []
    
    #Printing FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),3)
    
    results = pose.process(rgbImg)
    #print(results.pose_landmarks)       #Prints the keypoints
    
    if results.pose_landmarks:
        for id, kpt in enumerate(results.pose_landmarks.landmark):       #Store the kpts in variables
            #print(f'{id} ',results.pose_landmarks.landmark[1])     #Prints the ID and their kpts location
            imgHeight,imgWidth,conf = img.shape
            cx, cy = int(kpt.x*imgWidth), int(kpt.y*imgHeight)
            
            for i in range(len(results.pose_landmarks.landmark)):
                x = kpt.x
                y = kpt.y
                temp.append(x)
                temp.append(y)
                
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        prediction = model.predict([np.asarray(temp)])
        predictiedAction = labelsDict[prediction[0]]
        
        #print(predictiedAction)
        cv2.putText(img, predictiedAction, (20,80), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),3)
        
    cv2.imshow("Video Capture", img)
    if cv2.waitKey(10) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()