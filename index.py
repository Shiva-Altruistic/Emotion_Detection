from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import cv2
import base64

app = FastAPI()

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "emotion_mobilenet.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# FER-2013 labels
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

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- REQUEST SCHEMA ----------------
class ImageData(BaseModel):
    image: str  # base64 image

# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict(data: ImageData):
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(data.image.split(",")[1])
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return {
                "face": None,
                "emotion": "No face",
                "confidence": 0,
                "reply": "Camera image not clear.",
                "emoji": "⚠️"
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return {
                "face": None,
                "emotion": "No face",
                "confidence": 0,
                "reply": "Please face the camera clearly.",
                "emoji": "👤"
            }

        # Take first detected face
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]

        # FER-2013 preprocessing
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = face.reshape(1, 48, 48, 1)

        # ---------------- MODEL INFERENCE ----------------
        prediction = model.predict(face, verbose=0)[0]

        emotion_index = int(np.argmax(prediction))
        emotion = emotions[emotion_index]
        confidence = float(prediction[emotion_index] * 100)

        return {
            "face": {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            },
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "reply": responses[emotion],
            "emoji": emoji_map[emotion]
        }

    except Exception as e:
        return {
            "face": None,
            "emotion": "Error",
            "confidence": 0,
            "reply": "Processing error occurred.",
            "emoji": "❌"
        }
