import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging


# Load model once globally
model = load_model("emotion_detector_model.h5")

# Constants
IMG_HEIGHT = 48
IMG_WIDTH = 48
TIME_STEPS = 6
CHUNK_SIZE = 8  # So width 48 -> 6 chunks of 8
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # Change based on your dataset

def preprocess_frame(frame):
    # Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))  # 48x48
    norm_img = resized / 255.0

    # Split the width into 6 chunks (each 8 pixels wide)
    chunks = [norm_img[:, i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in range(TIME_STEPS)]
    sequence = np.stack([chunk[..., np.newaxis] for chunk in chunks], axis=0)  # (6, 48, 8, 1)

    return sequence[np.newaxis, ...]  # shape: (1, 6, 48, 8, 1)


def detect_cnn(image_path):
    # Read image
    logging.debug(f"Reading image from {image_path}")
    frame = cv2.imread(image_path)
    img = image.load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# ✅ Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

# ✅ Label map (check your training classes to confirm order)
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ✅ Print result
    # Encode image to base64 to send back
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    emotion=class_labels[predicted_class]
    return emotion,frame_base64
