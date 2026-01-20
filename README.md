🎭 Facial Emotion Recognition Web Application

This project presents an end-to-end real-time facial emotion recognition system built using deep learning and modern web technologies. A MobileNet-based CNN, trained on the FER-2013 dataset, is used to classify facial expressions into seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The system captures live webcam frames on the client side, detects faces using Haar Cascade, preprocesses facial regions to match training specifications, and performs inference through a FastAPI backend. The predicted emotion and its confidence score are returned and visualized in real time with bounding boxes.

The application is deployed on Vercel, enabling a scalable, serverless ML-powered web experience.

🔑 Features

Live webcam-based emotion detection

Face detection with bounding box visualization

Emotion classification with confidence scores

FER-2013 + MobileNet deep learning model

FastAPI backend for inference

Deployed on Vercel (HTTPS-enabled)

🛠 Tech Stack

Python, TensorFlow, OpenCV

FastAPI

HTML, JavaScript (Webcam API)

Vercel (Deployment)
