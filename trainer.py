import cv2
import numpy as np
from PIL import Image
import os

#path for dataset
path = './dataset'
recogniser = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

#gather the iamges and label data
def getImageNLabels(path):
    imgDir = [os.path.join(path,f) for f in os.listdir(path)]
    facesArr = []
    idArr = []
    for imagePath in imgDir :
        PIL_img = Image.open(imagePath).convert('L') #greyscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        #same as facedetection.py
        faces = detector.detectMultiScale(img_numpy)
        for(x,y,w,h) in faces:
            facesArr.append(img_numpy[y:y+h, x:x+w])
            idArr.append(id)
    return facesArr, idArr

print("\n Training faces")
x, y = getImageNLabels(path)

#saves model into yml
recogniser.train(x, np.array(y))
recogniser.write('./trainer/trainer.yml')

#print the number of faces trained and end program
print("\n {0} faces trained".format(np.unique(y)))


        
    