import os
import numpy as np
import cv2

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

haar_data = cv2.CascadeClassifier(haar_model)

print('Training without Mask')
capture = cv2.VideoCapture(0)
data_withoutmask = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            face = img[y:y+h, x:x+w,:]
            face = cv2.resize(face,(50,50))
            if len(data_withoutmask)<400:
                if len(data_withoutmask) == 1:
                    print('Please wait...')
                data_withoutmask.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27 or len(data_withoutmask) >= 200:
            break
capture.release()
cv2.destroyAllWindows()
np.save('Without_mask.npy',data_withoutmask)

print('Training with Mask')
capture = cv2.VideoCapture(0)
data_withmask = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            face = img[y:y+h, x:x+w,:]
            face = cv2.resize(face,(50,50))
            if len(data_withmask)<400:
                if len(data_withmask) == 1:
                    print('Please wait...')
                data_withmask.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27 or len(data_withmask) >= 200:
            break
capture.release()
cv2.destroyAllWindows()
np.save('With_mask.npy',data_withmask)
