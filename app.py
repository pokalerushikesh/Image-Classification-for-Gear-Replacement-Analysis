import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import requests
import os

# Hugging Face Model URL
MODEL_URL = "https://huggingface.co/rushikesh830/gear-classification-model/resolve/main/gear_classifier.pkl"
MODEL_PATH = "gear_classifier.pkl"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully!")

# Load the Pickle Model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

model.eval()  # Set to evaluation mode

# Define Class Labels
class_labels = ["rusty_gears", "undamaged_gears", "damaged_gears"]

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction Function
def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_labels[predicted.item()]

    # Maintenance Recommendation
    recommendation = ""
    if predicted_class == "rusty_gears":
        recommendation = "⚠️ Maintenance Required"
    elif predicted_class == "damaged_gears":
        recommendation = "❌ Replacement Required"
    else:
        recommendation = "✅ No Issues Found"

    return predicted_class, recommendation

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Text(label="Predicted Class"), gr.Text(label="Maintenance Recommendation")],
    title="Gear Classification",
    description="Upload an image of a gear to classify it as Rusty, Undamaged, or Damaged."
)

if __name__ == "__main__":
    iface.launch(share=True)
