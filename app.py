import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# Define model (must match your training)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*30*30, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*30*30)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Load model
model = CNNModel()
model.load_state_dict(torch.load("deep_cnn_model_weights.pth", map_location=torch.device('cpu')))
model.eval()

st.title("Deepfake Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img = np.array(image)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.transpose(img, (2,0,1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(img).item()

    if pred > 0.5:
        st.error("Deepfake Image")
    else:
        st.success("Real Image")
