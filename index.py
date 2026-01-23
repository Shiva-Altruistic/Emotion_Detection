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

# --- SETUP ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL ---
MODEL_PATH = "emotion_mobilenet.h5"
model = None

# Load model safely
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
else:
    print(f"❌ File not found: {MODEL_PATH}")

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emoji_map = {"Angry": "😡", "Disgust": "🤢", "Fear": "😨", "Happy": "😄", "Sad": "😢", "Surprise": "😲", "Neutral": "😐"}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class ImageData(BaseModel):
    image: str

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r") as f:
            return f.read()
    return "Error: static/index.html not found"

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
def predict(data: ImageData):
    if model is None:
        return {"emotion": "Error", "reply": "Model file not loaded on server", "emoji": "⚠️"}

    try:
        # 1. Decode Image
        img_str = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Detect Face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

        if len(faces) == 0:
            return {"face": None, "emotion": "No Face", "confidence": 0, "reply": "Looking for face...", "emoji": "👤"}

        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        # --- SMART PREDICTION (Auto-Detect Shape) ---
        
        # Prepare 1-Channel (Black & White) - Shape: (1, 48, 48, 1)
        resize_gray = cv2.resize(face_roi, (48, 48))
        norm_gray = resize_gray.astype("float32") / 255.0
        input_1ch = np.expand_dims(norm_gray, axis=0)
        input_1ch = np.expand_dims(input_1ch, axis=-1)

        # Prepare 3-Channel (Color) - Shape: (1, 48, 48, 3)
        # Stack the gray image 3 times to fake RGB
        input_3ch = np.stack((norm_gray,)*3, axis=-1)
        input_3ch = np.expand_dims(input_3ch, axis=0)

        preds = None
        
        # Try 1-Channel first (Standard for FER models)
        try:
            preds = model.predict(input_1ch, verbose=0)[0]
        except:
            # If that fails, try 3-Channel (Standard for MobileNet)
            try:
                preds = model.predict(input_3ch, verbose=0)[0]
            except Exception as final_err:
                # If both fail, send the ACTUAL error to the phone screen
                raise ValueError(f"Shape Error: {str(final_err)}")

        # Result
        idx = int(np.argmax(preds))
        conf = float(preds[idx] * 100)
        
        return {
            "face": [int(x), int(y), int(w), int(h)],
            "emotion": emotions[idx],
            "confidence": round(conf, 1),
            "reply": "Success!",
            "emoji": emoji_map.get(emotions[idx], "😐")
        }

    except Exception as e:
        # This will print the exact error on your phone screen
        return {
            "face": None,
            "emotion": "Error",
            "confidence": 0,
            "reply": str(e)[:100],  # Show first 100 chars of error
            "emoji": "❌"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
