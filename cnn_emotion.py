# cnn_emotion.py
import cv2
import numpy as np
import base64
from model import ERModel

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = ERModel("model.json", "model.weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def detect_emotion(image_path):
    fr = cv2.imread(image_path)
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)

    final_pred = ""
    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]

        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        final_pred = pred

        cv2.putText(fr, pred, (x, y - 10), font, 0.9, (0, 255, 0), 2)
        cv2.rectangle(fr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', fr)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return final_pred, frame_base64
