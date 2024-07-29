import cv2
import numpy as np
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('open_pose_project/Industrial_Project_HAR_for_Jetson_Nano/test_video/walking.mp4')

while True:
    success, img = cap.read()
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = pose.process(rgbImg)
    
    #print(results.pose_landmarks)       #Prints the keypoints
    
    if results.pose_landmarks:
        for id, kpt in enumerate(results.pose_landmarks.landmark):       #Store the kpts in variables
            imgHeight,imgWidth,conf = img.shape
            cx, cy = int(kpt.x*imgWidth), int(kpt.y*imgHeight)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    cv2.imshow("Video Capture", img)
    if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()