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
    
    # 1. Load the checkpoint
    checkpoint = torch.load(
        "model_checkpoint_deepcnn.pth", 
        map_location=device, 
        weights_only=False
    )

    # 2. Extract the actual state dict
    # If the file was saved as a dictionary of states, get 'model_state_dict'
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 3. Handle DataParallel prefix (Common issue)
    # If model was trained on GPU/DataParallel, keys start with 'module.'
    # This strips 'module.' so it matches your local CNNModel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # 4. Load weights into model
    model.load_state_dict(new_state_dict)
    
    model.to(device)
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
        # Since output is a tensor like [[0.85]], use .item()
        probability = output.item() 

    if probability >= 0.5:
        label = "Deepfake"
        conf = probability * 100
    else:
        label = "Real"
        conf = (1 - probability) * 100 # Confidence for 'Real'

    return label, conf

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
