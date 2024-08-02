import cv2
import mediapipe as mp
import time
import os
import pickle

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

fileName = 'open_pose_project/Industrial_Project_HAR_for_Jetson_Nano/test_video/walking.mp4'
cap = cv2.VideoCapture(fileName)
#cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
pTime = 0

actionData = []
actionName = []

def getFileName(name):
    out = os.path.basename(name)
    #print(os.path.splitext(out)[0])        #Checking output
    return os.path.splitext(out)[0]

while True:
    success, img = cap.read()
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    temp = []
    
    #Printing FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),3)
    
    results = pose.process(rgbImg)
    #print(results.pose_landmarks)       #Prints the keypoints
    
    if results.pose_landmarks:
        for id, kpt in enumerate(results.pose_landmarks.landmark):       #Store the kpts in variables
            #print(f'{id} ',results.pose_landmarks.landmark[1])     #Prints the ID and their kpts location
            imgHeight,imgWidth,conf = img.shape
            cx, cy = int(kpt.x*imgWidth), int(kpt.y*imgHeight)
            
            #Filling the arrays with data
            temp.append(kpt.x)
            temp.append(kpt.y)
            
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        #Storing the data
        actionData.append(temp)
        actionName.append(getFileName(fileName))
        
    cv2.imshow("Video Capture", img)
    if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        
    f = open('open_pose_project/Industrial_Project_HAR_for_Jetson_Nano/data/data.pickle', 'wb')
    pickle.dump({'actionData': actionData, 'actionName': actionName},f)
    f.close()      

cap.release()
cv2.destroyAllWindows()