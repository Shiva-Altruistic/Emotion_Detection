from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import cv2
import base64
import os

# --- 1. SETUP ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # CPU only
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD MODEL ---
MODEL_PATH = "emotion_mobilenet.h5"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
else:
    model = None
    print(f"❌ ERROR: Model {MODEL_PATH} not found.")

# Standard FER-2013 Labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

responses = {
    "Angry": "Take a deep breath.",
    "Disgust": "Something unpleasant?",
    "Fear": "You are safe here.",
    "Happy": "Yay! Keep smiling! 😄",
    "Sad": "Sending you a virtual hug.",
    "Surprise": "Wow! What happened?",
    "Neutral": "You look focused."
}

emoji_map = {
    "Angry": "😡", "Disgust": "🤢", "Fear": "😨",
    "Happy": "😄", "Sad": "😢", "Surprise": "😲", "Neutral": "😐"
}

# Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class ImageData(BaseModel):
    image: str

# --- 3. ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
def predict(data: ImageData):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        # Decode Image
        img_str = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Resize entire image to 320x320 (Matches Frontend)
        img = cv2.resize(img, (320, 320))
        
        # 2. Convert to Gray for Face Detection (Haar works best on Gray)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Detect Face (More sensitive settings)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4, 
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return {
                "face": None,
                "emotion": "No Face",
                "confidence": 0,
                "reply": "Waiting for a face...",
                "emoji": "👤"
            }

        # 4. Crop Face
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w] 

        # --- KEY FIX FOR MOBILENET ---
        # Resize to 48x48
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Normalize (0 to 1)
        face_roi = face_roi.astype("float32") / 255.0
        
        # Expand dims to (1, 48, 48)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Convert to 3 Channels (RGB) for MobileNet
        # We stack the grayscale image 3 times: shape becomes (1, 48, 48, 3)
        face_roi = np.stack((face_roi,)*3, axis=-1)
        
        # Note: If your model specifically wants (1, 48, 48, 1), 
        # change the line above to: face_roi = np.expand_dims(face_roi, axis=-1)
        # But MobileNet usually needs 3 channels.
        face_roi = face_roi.reshape(1, 48, 48, 3)

        # 5. Predict
        preds = model.predict(face_roi, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx] * 100)
        emotion = emotions[idx]

        return {
            "face": [int(x), int(y), int(w), int(h)],
            "emotion": emotion,
            "confidence": round(confidence, 1),
            "reply": responses.get(emotion, ""),
            "emoji": emoji_map.get(emotion, "😐")
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"emotion": "Error", "reply": "Server Error", "emoji": "❌"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
