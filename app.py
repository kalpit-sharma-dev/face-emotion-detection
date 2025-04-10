from flask import Flask, request, jsonify
from flask_cors import CORS
from model.clip_emotion import detect_emotion
import cv2
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "temp_frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
