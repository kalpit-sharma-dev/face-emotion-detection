from flask import Flask, request, jsonify , render_template , Response
from flask_cors import CORS
from models.clip_emotion import detect_emotion
import cv2
from camera import VideoCamera
import os

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

@app.route('/cnnkeras')
def index_cnn():
    return render_template('cnnkeras.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/analyze", methods=["POST"])
def analyze():
    if "frame" not in request.files:
        return jsonify({"error": "No frame received"}), 400

    file = request.files["frame"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        emotion, scores = detect_emotion(filepath)
        return jsonify({"emotion": emotion, "scores": scores})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
