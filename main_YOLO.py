from ultralytics import YOLO 

#Loading pretrained model
model = YOLO('./model/yolov8n-pose.pt')

sourceImg = './test_image/image.jpg'

#Output
results = model(sourceImg, show=True, conf=0.3, save=True)