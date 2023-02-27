import os
import cv2 as cv
import numpy as np

people = list()
for i in os.listdir(r"Basics\Photos\Faces\train"):
    people.append(i)

haar_cascade = cv.CascadeClassifier('Basics\Code\haar_face.xml')
DIR = r'Basics\Photos\Faces\train'

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recogonizer on the features list and the labels list

face_recognizer.train(features, labels)
print('Training Done ------------------')

np.save('features.npy',features)
np.save('labels.npy', labels)

face_recognizer.save('face_trained.yml')