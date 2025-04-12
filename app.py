from flask import Flask, request, jsonify , render_template , Response
from flask_cors import CORS
from models.clip_emotion import detect_emotion , detect_age
import cv2
from camera import VideoCamera
import os
from cnn_emotion import detect_emotion as detect_emotion_cnn
# from cnn_lstm import detect_cnn_lstm_emotion
from cnn_resnet import detect_cnn_resnetemotion

from cnn import detect_cnn

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO in production
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "temp_frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/vit')
def index_vit():
    return render_template('vit.html')

@app.route('/cnn')
def index_cnnonly():
    return render_template('cnn.html')

@app.route('/cnnkeras')
def index_cnn():
    return render_template('cnnkeras.html')

@app.route('/cnnlstm')
def index_cnnlstm():
    return render_template('cnn_lstm.html')

@app.route('/cnn_resnet')
def index_cnn_resnet():
    return render_template('cnn_resnet.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed', methods=["POST"])
def video_feed():
    # return Response(gen(VideoCamera()),
    #                 mimetype='multipart/x-mixed-replace; boundary=frame')
    
    if "frame" not in request.files:
        return jsonify({"error": "No frame received"}), 400

    file = request.files["frame"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    try:
        emotion, image_base64 = detect_emotion_cnn(filepath)
        return jsonify({"emotion": emotion, "image": image_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    if "frame" not in request.files:
        return jsonify({"error": "No frame received"}), 400

    file = request.files["frame"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        emotion, scores = detect_emotion(filepath)
        age, age_scores = detect_age(filepath)
        return jsonify({"emotion": emotion, "scores": scores, "age": age})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cnn_lstm_video_feed', methods=["POST"])
def cnn_lstm_video_feed():
    
    if "frame" not in request.files:
        return jsonify({"error": "No frame received"}), 400

    file = request.files["frame"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    try:
        emotion, image_base64 = detect_emotion_cnn(filepath)
        return jsonify({"emotion": emotion, "image": image_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cnn_resnet', methods=['POST'])
def cnn_resnet():
    if "frame" not in request.files:
        logging.warning("No frame in request")
        return jsonify({"error": "No frame received"}), 400
    file = request.files["frame"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    logging.info(f"File saved to {filepath}")
    file.save(filepath)
    try:
        emotion, image_base64 = detect_cnn_resnetemotion(filepath)
        return jsonify({"emotion": emotion, "image": image_base64})
    except Exception as e:
        logging.error(f"Failed to save file: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/cnn', methods=['POST'])
def cnn():
    if "frame" not in request.files:
        logging.warning("No frame in request")
        return jsonify({"error": "No frame received"}), 400
    file = request.files["frame"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    logging.info(f"File saved to {filepath}")
    file.save(filepath)
    try:
        emotion, image_base64 = detect_cnn(filepath)
        return jsonify({"emotion": emotion, "image": image_base64})
    except Exception as e:
        logging.error(f"Failed to save file: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
