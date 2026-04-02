# Deepfake Image Detection using CNN

## 📌 Project Overview
This project detects whether an image is **Real** or **Deepfake** using a Convolutional Neural Network (CNN).

## 🎯 Features
- Upload image for detection
- Real-time prediction
- Confidence score output
- Simple web interface

## 🛠 Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit

## 🧠 Model Details
- CNN-based architecture
- Binary classification (Real = 0, Deepfake = 1)
- Image size: 128x128
- Loss: Binary Crossentropy
- Optimizer: Adam

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
