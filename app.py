import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# Load Class Labels
class_labels = ["rusty_gears", "undamaged_gears", "damaged_gears"]

# Load the full model (architecture + weights)
model = torch.load("gear_classifier_finetuned.pth", map_location=torch.device("cpu"))
model.eval()  # Set to evaluation mode

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),         # Convert image to tensor
])

# Streamlit UI
st.title("Gear Classification App")
st.write("Upload an image of a gear to classify it as **Damaged**, **Rusty**, or **Undamaged**.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Ensure the image has 3 channels (convert to RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
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

    # Add Maintenance/Replacement Suggestions
    if predicted_class == "rusty_gears":
        st.write("### **Maintenance Required**: The gear shows signs of rust and needs maintenance.")
    elif predicted_class == "damaged_gears":
        st.write("### **Replacement Required**: The gear is damaged and should be replaced.")
    else:
        st.write("### **No Action Needed**: The gear is in good condition.")
