import cv2
import mediapipe as mp
import os
import pickle

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

fileName = './test_video/walking.mp4'
cap = cv2.VideoCapture(fileName)

cap.set(3,640)
cap.set(4,480)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)


actionData = []
actionName = []
temp = []

def getFileName(name):
    out = os.path.basename(name)
    #print(os.path.splitext(out)[0])        #Checking output
    return os.path.splitext(out)[0]

while True:
    success, img = cap.read()
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    results = pose.process(rgbImg)
    #print(results.pose_landmarks)       #Prints the keypoints
    
    if results.pose_landmarks:
        for id, kpt in enumerate(results.pose_landmarks.landmark):       #Store the kpts in variables
            #print(f'{id} ',results.pose_landmarks.landmark[1])     #Prints the ID and their kpts location
            imgHeight,imgWidth,conf = img.shape
            cx, cy = int(kpt.x*imgWidth), int(kpt.y*imgHeight)
            
            for i in range(len(results.pose_landmarks.landmark)):
                x = kpt.x/int(cv2.CAP_PROP_FRAME_HEIGHT)
                y = kpt.y/int(cv2.CAP_PROP_FRAME_WIDTH)
                temp.append(x)
                temp.append(y)
            
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        #Storing the data
        actionData.append(temp)
        actionName.append(getFileName(fileName))
        
    cv2.imshow("Video Capture", img)
    if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        
    f = open('./data/data.pickle', 'wb')
    pickle.dump({'actionData': actionData, 'actionName': actionName},f)
    f.close()      

cap.release()
cv2.destroyAllWindows()