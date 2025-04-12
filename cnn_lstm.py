# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import base64

# # Load the model once
# model = load_model("emotion_cnn_lstm.h5")

# # Emotion labels
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # Load Haar cascade
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def detect_cnn_lstm_emotion(image_path):
#     frame = cv2.imread(image_path)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#     print(f"Detected faces: {len(faces)}")

#     final_pred = "No face detected"

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         roi = roi_gray.astype("float") / 255.0
#         roi = img_to_array(roi)
#         roi = np.expand_dims(roi, axis=0)

#         prediction = model.predict(roi)[0]
#         label = emotion_labels[np.argmax(prediction)]
#         final_pred = label

#         # Draw results on original image
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Encode frame to base64
#     _, buffer = cv2.imencode('.jpg', frame)
#     frame_base64 = base64.b64encode(buffer).decode('utf-8')

#     return final_pred, frame_base64
