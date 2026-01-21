from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import cv2
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- LOAD MODEL ----------
MODEL_PATH = "emotion_mobilenet.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

MODEL_ACCURACY = 87.6

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

responses = {
    "Angry": "Please calm down.",
    "Disgust": "Something feels unpleasant.",
    "Fear": "Relax, you are safe.",
    "Happy": "Nice! Keep smiling 😄",
    "Sad": "It’s okay. Everything will be fine.",
    "Surprise": "That was unexpected!",
    "Neutral": "You look calm."
}

emoji_map = {
    "Angry": "😡",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐"
}

# ---------- FACE DETECTOR ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class ImageData(BaseModel):
    image: str

@app.post("/predict")
def predict(data: ImageData):
    try:
        img_bytes = base64.b64decode(data.image.split(",")[1])
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return {"face": None, "emotion": "No face", "confidence": 0,
                    "accuracy": MODEL_ACCURACY, "reply": "Camera unclear", "emoji": "⚠️"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

        if len(faces) == 0:
            return {"face": None, "emotion": "No face", "confidence": 0,
                    "accuracy": MODEL_ACCURACY, "reply": "Please face camera", "emoji": "👤"}

        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))

        preds = model.predict(face, verbose=0)[0]
        idx = int(np.argmax(preds))
        emotion = emotions[idx]
        confidence = float(preds[idx] * 100)

        return {
            "face": [int(x), int(y), int(w), int(h)],
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "accuracy": MODEL_ACCURACY,
            "reply": responses[emotion],
            "emoji": emoji_map[emotion]
        }

    except:
        return {"face": None, "emotion": "Error", "confidence": 0,
                "accuracy": MODEL_ACCURACY, "reply": "Processing error", "emoji": "❌"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 8000)))
