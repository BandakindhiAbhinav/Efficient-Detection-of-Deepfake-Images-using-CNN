import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import CNNModel

# Page Config
st.set_page_config(page_title="Deepfake Detection", page_icon="🧠", layout="centered")

st.title("Deepfake Image Detection")
st.write("Upload an image to verify its authenticity.")

device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = CNNModel()
    
    # Use the filename you downloaded
    checkpoint_path = "deep_cnn_model_weights.pth" 
    
    # 1. Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 2. Extract the weights (model_state_dict) from the checkpoint
    # If the file is just weights, it uses the checkpoint itself
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 3. Strip 'module.' prefix (crucial for Colab/Kaggle models)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 4. Load into model
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

model = load_model()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_image(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = model(img_tensor).item()

    # Labels based on alphabetical order: 0: Fake, 1: Real
    if probability >= 0.5:
        return "Real", probability * 100
    else:
        return "Deepfake", (1 - probability) * 100

# UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        label, confidence = predict_image(image)
        if label == "Deepfake":
            st.error(f"Result: {label} ({confidence:.2f}% confidence)")
        else:
            st.success(f"Result: {label} ({confidence:.2f}% confidence)")
