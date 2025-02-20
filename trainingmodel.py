'''In this code we can trian the dataset ,we can arrange the face data with halp of numpy,after the np.array of face data stored in yml'''
import cv2
import numpy as np
from PIL import Image
import os

recognzier=cv2.face.LBPHFaceRecognizer_create()

path="datasets"

def getImageID(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagePaths in imagePath:
        faceImage=Image.open(imagePaths).convert('L')
        faceNp=np.array(faceImage)
        Id=(os.path.split(imagePaths)[-1].split(".")[1])
        Id=int(Id)
        faces.append(faceNp)
        ids.append(Id)
        cv2.imshow("Trainimg",faceNp)
        cv2.waitKey(1)

    return ids,faces

Ids,facedata = getImageID(path)
recognzier.train(facedata, np.array(Ids))
recognzier.write("Trainer.yml")
cv2.destroyAllWindows()
print("--------Training Completed----------")