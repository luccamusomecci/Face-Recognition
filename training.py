import cv2
import os
import numpy as np

dataPath = 'C:/Users/luxaa/Desktop/Face Recognition/Data'
peopleList = os.listdir(dataPath)
print('[+] Person list: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('[+] Reading images ...')

    for fileName in os.listdir(personPath):
        print('[+] Faces: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName,0))
        image = cv2.imread(personPath + '/' + fileName,0)

    label = label + 1


face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print('[+] Training...')
face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modelLBPH.xml')
print('[+] Model has been saved.')