import cv2
import numpy as np

cap = cv2.VideoCapture(0)

scale = 320
confThreshold = 0.5     #Minimum threshold for detections
nmsThreshold = 0.3      #Threshold for boxes (Lower = more aggressive)

classFile = 'open_pose_project\Industrial_Project_HAR_for_Jetson_Nano\coco\coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().strip('\n').split('\n')


modelConfig = 'open_pose_project\Industrial_Project_HAR_for_Jetson_Nano\model\yolov3-tiny.cfg'
modelWeights = 'open_pose_project\Industrial_Project_HAR_for_Jetson_Nano\model\yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(out, camera):
    height, width, channel = camera.shape
    boundingBox = []    #contains x,y,w,h
    id = []     #containts the class ids
    confidence = []     #contains the confidence value
    
    for output in out:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)      #Tries to find the max value
            conf = scores[classId]     #Saves the value of the confidence
            if conf > confThreshold :
                w,h = int(detection[2]*width), int(detection[3]*height)
                x,y = int((detection[0]*width)-w/2), int((detection[1]*height)-h/2)
                boundingBox.append([x,y,w,h])
                id.append(classId)
                confidence.append(float(conf))
    nmsOut = cv2.dnn.NMSBoxes(boundingBox, confidence, confThreshold,nmsThreshold)
    for i in nmsOut:
        i = i
        box = boundingBox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(camera,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(camera,f'{classNames[id[i]].upper()} {int(confidence[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
                
while cap.isOpened():
    success, camera = cap.read()
    blob = cv2.dnn.blobFromImage(camera,1/255,(scale,scale),[0,0,0],1,crop=False)
    net.setInput(blob)
    
    #Getting layer names for referencing
    layerNames = net.getLayerNames()
    outputName = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    
    out = net.forward(outputName)
    #print(out[0].shape)     #First output layer; 300 bounding boxes
    #print(out[1].shape)     #Second output layer; 1200 bounding boxes
    
    #print(out[0][0])
    
    findObjects(out, camera)

    cv2.imshow("Video Capture", camera)
    if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()