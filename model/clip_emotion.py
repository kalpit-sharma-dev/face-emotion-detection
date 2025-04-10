import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

EMOTION_LABELS = ["happy", "sad", "angry", "surprised", "neutral", "disgusted", "fearful"]

def detect_emotion(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"A face showing {emotion}") for emotion in EMOTION_LABELS]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        logits_per_image, _ = model(image, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()

    result = dict(zip(EMOTION_LABELS, probs.tolist()))
    top_emotion = max(result, key=result.get)
    return top_emotion, result
