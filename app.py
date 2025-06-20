import streamlit as st
import requests
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import pipeline, set_seed

# Load GPT-2 text generator
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

generator = load_generator()

# Title
st.title("🖼️ Image Classification with Text Summary")

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
label = [line.strip() for line in requests.get(LABELS_URL).text.splitlines()]

# Image preprocessing function
def imagepro(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Classification function
def labe(image, model):
    global predicted
    global topic
    with st.spinner("🔍 Processing..."):
        i_tensor = imagepro(image)
        with torch.no_grad():
            outputs = model(i_tensor)
            _, predicted = torch.max(outputs, 1)
    topic = label[predicted.item()]

# Prediction display
def predict(topic_label):
    st.success(f"🎯 Predicted Item: **{topic_label}**")

# Text generation
def gen(topic, max_len):
    set_seed(42)
    result = generator(topic, max_length=max_len, num_return_sequences=1)
    st.success("✅ Text Generation Complete!")
    st.write("### 📝 Generated Text:")
    st.write(result[0]["generated_text"])

# UI Elements
upload = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
max_len = st.slider("✏️ Max Length of Generated Text", 50, 300, 100, step=10)
gun = st.button("🚀 Generate Text")

if upload is not None:
    image = Image.open(upload).convert('RGB')
    image = image.resize((224, 224))
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=False)

    # Run classification
    labe(image, model)
    predict(topic)

    # Generate text on button click
    if gun:
        gen(topic, max_len)
