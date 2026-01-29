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
import uvicorn
import urllib.request # Added to download the missing XML

# --- SETUP ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. AUTO-DOWNLOAD MISSING XML ---
xml_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_file):
    print(f"⬇️ Downloading {xml_file}...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    try:
        urllib.request.urlretrieve(url, xml_file)
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Failed to download XML: {e}")
        print("Please download 'haarcascade_frontalface_default.xml' manually.")

# Load Face Cascade from the local file we just checked/downloaded
face_cascade = cv2.CascadeClassifier(xml_file)

# --- 2. LOAD EMOTION MODEL ---
MODEL_PATH = "emotion_mobilenet.h5"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
else:
    print(f"❌ File not found: {MODEL_PATH}")

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emoji_map = {
    "Angry": "😡", "Disgust": "🤢", "Fear": "😨", "Happy": "😄", 
    "Sad": "😢", "Surprise": "😲", "Neutral": "😐"
}

class ImageData(BaseModel):
    image: str

# --- ROUTES ---
# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "Error: static/index.html not found."

@app.post("/predict")
def predict(data: ImageData):
    if model is None:
        return {"emotion": "Error", "reply": "Model not loaded", "emoji": "⚠️", "confidence": 0, "accuracy": 0}

    try:
        # Decode Image
        img_str = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect Face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

        if len(faces) == 0:
            return {"face": None, "emotion": "No Face", "confidence": 0, "reply": "Looking for face...", "emoji": "👤", "accuracy": 0}

        x, y, w, h = faces[0]
        face_roi = img[y:y+h, x:x+w] # Crop from Color Image

        # --- DYNAMIC SHAPE HANDLING ---
        input_shape = model.input_shape
        target_h = input_shape[1] if input_shape[1] else 224
        target_w = input_shape[2] if input_shape[2] else 224
        
        resized_face = cv2.resize(face_roi, (target_w, target_h))
        normalized_face = resized_face.astype("float32") / 255.0
        final_input = np.expand_dims(normalized_face, axis=0)

        # Predict
        preds = model.predict(final_input, verbose=0)[0]
        
        idx = int(np.argmax(preds))
        conf = float(preds[idx] * 100)
        predicted_emotion = emotions[idx]

        return {
            "face": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}, # Fixed Object Format
            "emotion": predicted_emotion,
            "confidence": round(conf, 1),
            "accuracy": 92,
            "reply": "Emotion Detected!",
            "emoji": emoji_map.get(predicted_emotion, "😐")
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "face": None, "emotion": "Error", "confidence": 0, 
            "reply": f"Error: {str(e)[:50]}...", "emoji": "❌", "accuracy": 0
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
