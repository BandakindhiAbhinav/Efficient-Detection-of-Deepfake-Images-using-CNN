import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import CNNModel

st.title("Deepfake Detection using CNN")

device = torch.device("cpu")

model = CNNModel()

model.load_state_dict(
    torch.load(
        "deep_cnn_model_weights.pth",
        map_location=device
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image)

    img = transform(image)

    img = img.unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        prediction = output.item()

    if prediction >= 0.5:

        st.success("Real Image")

    else:

        st.error("Fake Image")
