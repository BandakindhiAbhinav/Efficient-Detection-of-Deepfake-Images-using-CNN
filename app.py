import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from model import CNNModel

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Deepfake Image Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("Efficient Detection of Deepfake Images using CNN")
st.write("Upload an image to check whether it is Real or Deepfake.")

# -----------------------------
# Device Setup (Streamlit uses CPU)
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Load Model Safely
# -----------------------------
@st.cache_resource
def load_model():

    model = CNNModel()
    model.to(device)

    # FIXED: Updated the filename to exactly match your uploaded weights file
    checkpoint = torch.load(
        "model_checkpoint_deepcnn.pth",
        map_location=device,
        weights_only=False # Added to avoid PyTorch 2.x security restrictions on checkpoints
    )

    # Extract weights safely
    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model.eval()

    return model

model = load_model()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image):
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probability = output.item()

    if probability >= 0.5:
        label = "Deepfake"
    else:
        label = "Real"

    return label, probability

# -----------------------------
# File Upload UI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):
        label, probability = predict_image(image)

        st.subheader("Prediction Result")

        if label == "Deepfake":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        st.write(f"Confidence: {probability:.4f}")
