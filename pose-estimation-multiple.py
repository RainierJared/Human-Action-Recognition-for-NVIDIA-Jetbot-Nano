import cv2
import mediapipe as mp
import os
import pickle

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

actionData = []
actionName = []

DATA_DIR = './videos'
for dir_ in os.listdir(DATA_DIR):
    for path in os.listdir(os.path.join(DATA_DIR, dir_)):

        cap = cv2.VideoCapture(os.path.join(DATA_DIR,dir_,path))
        cap.set(3,640)
        cap.set(4,480)
        
        while True:
            success, img = cap.read()
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            temp = []

            results = pose.process(rgbImg)
            #print(results.pose_landmarks)       #Prints the keypoints
            
            if results.pose_landmarks:
                for id, kpt in enumerate(results.pose_landmarks.landmark):       #Store the kpts in variables
                    #print(f'{id} ',results.pose_landmarks.landmark[1])     #Prints the ID and their kpts location
                    imgHeight,imgWidth,conf = img.shape
                    cx, cy = int(kpt.x*imgWidth), int(kpt.y*imgHeight)
                    
                    for i in range(len(results.pose_landmarks.landmark)):
                        x = cx
                        y = cy
                        temp.append(x)
                        temp.append(y)
                    
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                #Storing the dataq
                actionData.append(temp)
                actionName.append(dir_)
                
            cv2.imshow("Video Capture", img)
            if cv2.waitKey(10) & 0xFF==ord('q'):
                    break
        
        cv2.waitKey(10)
            
    f = open('./data/data.pickle', 'wb')
    pickle.dump({'actionData': actionData, 'actionName': actionName},f)
    f.close()      

cap.release()
cv2.destroyAllWindows()