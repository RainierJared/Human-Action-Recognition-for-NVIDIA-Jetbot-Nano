import cv2
import numpy as np
from PIL import Image
import os
import dlib

#path for dataset
path = "face_detection_hog/dataset"
recogniser = cv2.face.LBPHFaceRecognizer_create()
detector = dlib.get_frontal_face_detector()

#gather dataset
def getImages(path):
    imgDir = [os.path.join(path,f) for f in os.listdir(path)]
    facesArr = []
    idArr = []
    for imagePath in imgDir:
        PIL_img = Image.open(imagePath).convert('L')
        img_np = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        #HOG detection model
        face = detector(img_np)
    
        for(i, rect) in enumerate(face):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            facesArr.append(img_np[y:y+h, x:x+w])
            idArr.append(id)
        
    return facesArr, idArr

print("\n Training faces")
x, y = getImages(path)

#saves model into yml
recogniser.train(x, np.array(y))
recogniser.write('face_detection_hog/trainer/trainer.yml')

#print the number of faces trained and end program
print("\n {0} faces trained".format(np.unique(y)))