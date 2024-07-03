from ultralytics import YOLO
import cv2
import numpy as np
import sys


# #Loading pretrained model **ONLY NEED TO RUN IT ONCE AT THE START TO CREATE THE LIGHTWEIGHT MODEL
# model = YOLO('./open_pose_project/Industrial_Project_HAR_for_Jetson_Nano/model/yolov8n-pose.pt')
# model.export(format='ncnn')    #creates './yolov8n_ncnn_model'

ncnn_model=YOLO('./open_pose_project/Industrial_Project_HAR_for_Jetson_Nano/model/yolov8n-pose_ncnn_model')

camera = cv2.VideoCapture("./open_pose_project/Industrial_Project_HAR_for_Jetson_Nano/test_video/walking.mp4")     #For videos when training the classifier
#camera = cv2.VideoCapture(0)        #For live feed

if (camera.isOpened() == False):
    print("Error opening camera")
    sys.exit()
    
while camera.isOpened():
    success, frame = camera.read()
    if success:
        results = ncnn_model.predict(frame, verbose=False)
        keyPoints = results[0].keypoints.data
        for kpts in keyPoints:
            for pts in kpts:
                # cv2.circle(frame, (int(pts[0]), int(pts[1])), 4, (0,255,0), -1)       #For drawing the keypoints
                print(f'X: {pts[0]}, Y: {pts[1]}, Z: {pts[2]}')

        cv2.imshow("Camera",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
camera.release()
cv2.destroyAllWindows()

