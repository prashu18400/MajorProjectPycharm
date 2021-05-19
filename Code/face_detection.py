import cv2
import numpy as np
# Load the cascade
from pretrained_model import model_load
from Predict import predict

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
model = model_load.model_load()
model.summary()
b = True
while b:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    # Draw the rectangle around each face
    try:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            crop_face = img[y:y + h, x:x + w]
        # Display
        a = predict.pre_process(crop_face)
        cv2.imshow('img', img)
        cv2.imshow('face', crop_face)
        # filename = 'save.jpg'
        # cv2.imwrite(filename, crop_face)
    # Stop if escape key is pressed
    except Exception:
        pass

    prediction = model.predict(np.array([a/255]))
    sex_f = ['Male',"Female"]
    age = int(np.round(prediction[1][0]))
    sex = int(np.round(prediction[0][0]))
    # img = cv2.putText(img,str(age), cv2.FONT_HERSHEY_SIMPLEX,
    #                1, (255,0,0), 2, cv2.LINE_AA)
    # print(age,sex_f[sex])
    b = False
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    Age = str(age)+','+(sex_f[sex])
img = cv2.putText(img,str(Age),(450,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
cv2.imshow('image',img)
cv2.imwrite('final.jpg',img)
cv2.waitKey(5000)
# Release the VideoCapture object
cap.release()
