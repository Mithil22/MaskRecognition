import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

haar_data = cv2.CascadeClassifier(haar_model)

with_mask = np.load('With_mask.npy')
without_mask = np.load('Without_mask.npy')

with_mask = with_mask.reshape(200, 50*50*3)
without_mask = without_mask.reshape(200, 50*50*3)

X = np.r_[with_mask, without_mask]

labels = np.zeros(X.shape[0])
labels[200:] = 1
output = {0 : 'Mask', 1 : 'No Mask'}

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2)

pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2)
svm = SVC()
svm.fit(X_train, y_train)


capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            face = img[y:y+h, x:x+w,:]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            pred = svm.predict(face)[0]
            op = output[int(pred)]
            cv2.putText(img, op, (x,y), font, 1 , (244,156,143),2)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27:
            break
capture.release()
cv2.destroyAllWindows()
