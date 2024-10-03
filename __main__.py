import cv2
import mediapipe as mp
import numpy as np
import time
import datetime
import csv
import pickle

#The different actions recognised
labelsDict = {0: 'sitting', 1: 'moving', 2: 'standing', 3: 'laying down'}

#Directory for test video
fileName = './test-videos/sitting-test.mp4'

#Loading the models
model_dict = pickle.load(open('./model/model.p', 'rb'))
model = model_dict['model']
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=0)

#Set the video capture and dimensions (0 for camera)
#cap = cv2.VideoCapture(camera(flip_method=0),cv2.CAP_GSTREAMER)     #For nano's camera
cap=cv2.VideoCapture(fileName)
cap.set(3,640)
cap.set(4,480)

#Gathering dimensions
width = int(cap.get(3))
height = int(cap.get(4))
size = (width, height)

#For FPS
pTime=0

#For log
pOut="."
cOut="."
confidence=["0000"]
#For recording
#result = cv2.VideoWriter('./out-video/result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size, True)

#Prints FPS
def printFPS():
    global pTime
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    return int(fps)

#The action recognition
def actionRecognition():
    global pOut, cOut, confidence
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    temp = []
    
    #Printing FPS
    cv2.putText(img, f'FPS: {printFPS()}', (20,40), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0),2)
    
    #Gathers keypoints
    results = pose.process(rgbImg)
    #print(results.pose_landmarks)       #Prints the keypoints
    
    if results.pose_landmarks:

        for id, kpt in enumerate(results.pose_landmarks.landmark):       #Store the kpts in variables
            #print(f'{id} ',results.pose_landmarks.landmark[1])     #Prints the ID and their kpts location
            imgHeight,imgWidth,_ = img.shape
            cx, cy = int(kpt.x*imgWidth), int(kpt.y*imgHeight)
            for i in range(len(results.pose_landmarks.landmark)):
                x = cx
                y = cy
                temp.append(x)
                temp.append(y)
            
        
        #Prints landmarks
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
        #Prints the prediction
        prediction = model.predict([np.asarray(temp)])
        confidence = model.predict_proba([np.asarray(temp)])        
        cOut = predictedAction(prediction)
        cv2.putText(img, cOut, (20,70), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255),2)
        #result.write(img)
    if cOut != pOut:
        logToCSV(cOut)
            
    pOut=cOut

def logToCSV(out):
    rows = [
        {'Date': dateAndTime(),
         'Action': out}
    ]
    with open('./log/log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(rows)
        
def predictedAction(prediction):
    return labelsDict[int(prediction[0])]

def dateAndTime():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
 
#Starts the file
def start():
    global img, confidence
    while True:
        success, img = cap.read()
        if success:
            actionRecognition()
            cv2.putText(img, f'Sitting: {confidence[0][0]}', (20,100), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0),2)
            cv2.putText(img, f'Moving: {confidence[0][1]}', (20,130), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0),2)
            cv2.putText(img, f'Standing: {confidence[0][2]}', (20,160), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0),2)
            cv2.putText(img, f'Laying Down: {confidence[0][3]}', (20,190), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0),2)
            cv2.imshow("Video Capture", img)   
        else:
            break  
        
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

if __name__ == "__main__":
    start()
    
cap.release()
#result.release()
cv2.destroyAllWindows()