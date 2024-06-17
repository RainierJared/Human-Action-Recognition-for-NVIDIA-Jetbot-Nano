from ultralytics import YOLO

#Loading pretrained model
model = YOLO('./open_pose_project/Industrial_Project_HAR_for_Jetson_Nano/model/yolov8n-pose.pt')

model.export(format='ncnn')    #creates './yolov8n_ncnn_model'

ncnn_model=YOLO('./open_pose_project/Industrial_Project_HAR_for_Jetson_Nano/model/yolov8n-pose_ncnn_model')

results = ncnn_model(source=0, show=True, conf=0.3, save=True)