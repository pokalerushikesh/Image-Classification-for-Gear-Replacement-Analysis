import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import gdown
import os

# Define the model path and download if not present
MODEL_PATH = "gear_classifier_finetuned.pth"
MODEL_URL = "https://drive.google.com/uc?id=1mruQYU_iGBIG0pC2M778AyI8Nixe8iPx"  # Direct Google Drive link

# Download the model from Google Drive if not found locally
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model file...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the Entire Model
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.eval()  # Set to evaluation mode

# Define Class Labels
class_labels = ["rusty_gears", "undamaged_gears", "damaged_gears"]

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("Gear Classification App")
st.write("Upload an image of a gear to classify it as **Damaged**, **Rusty**, or **Undamaged**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure it's a 3-channel image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess Image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make Prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_labels[predicted.item()]

    # Display Prediction
    st.write(f"### Prediction: **{predicted_class}**")

    # Show Maintenance Recommendation
    if predicted_class == "rusty_gears":
        st.warning("⚠️ **Maintenance Required**: The gear appears to be rusted. Regular maintenance is recommended.")
    elif predicted_class == "damaged_gears":
        st.error("❌ **Replacement Required**: The gear is damaged and may need replacement.")
    else:
        st.success("✅ **No Issues Detected**: The gear appears to be in good condition.")
