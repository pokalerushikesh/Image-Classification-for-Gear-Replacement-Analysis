import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import os
import gdown 

# Define Google Drive file ID and local model path
GOOGLE_DRIVE_FILE_ID = "1mruQYU_iGBIG0pC2M778AyI8Nixe8iPx"
MODEL_PATH = "gear_classifier_finetuned.pth"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

# Load Class Labels
class_labels = ["rusty_gears", "undamaged_gears", "damaged_gears"]

# Define Model Architecture (ResNet18 with 3 output classes)
class GearClassifier(torch.nn.Module):
    def __init__(self):
        super(GearClassifier, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, len(class_labels))

    def forward(self, x):
        return self.model(x)

# Download model if not available
download_model()

# Load the model
model = GearClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

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
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to RGB if it has an alpha channel (4th channel)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Preprocess Image
    image = transform(image).unsqueeze(0)

    # Make Prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_labels[predicted.item()]

    # Display Result
    st.write(f"### Prediction: **{predicted_class}**")

    # Show Maintenance Recommendations
    if predicted_class == "rusty_gears":
        st.warning("⚠️ Maintenance Required: Consider lubrication and rust removal.")
    elif predicted_class == "damaged_gears":
        st.error("❌ Replacement Required: The gear is severely damaged.")
    else:
        st.success("✅ No Issues: The gear is in good condition!")
