import streamlit as st
import requests
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import pipeline, set_seed

# Load GPT-2 text generator once
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

generator = load_generator()

# Load pretrained image classification model
model = models.resnet18(pretrained=True)
model.eval()

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = [line.strip() for line in requests.get(LABELS_URL).text.splitlines()]

# Preprocess image
def imagepro(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

st.title("🖼️ Image Classification + Text Summary App")

# Upload image
upload = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
max_len = st.slider("✏️ Max Length of Generated Text", 50, 300, 100, step=10)
generate_btn = st.button("🚀 Generate Text")

# If image is uploaded
if generate_btn :
    if upload is None:
        st.info(" Please First The Uplaod IMage")
    else:
        image = Image.open(upload).convert('RGB')
        image = image.resize((200, 200))
        with st.spinner("Processing..."):
            i_tensor = imagepro(image)
            with torch.no_grad():
                outputs = model(i_tensor)
                a, predicted = torch.max(outputs, 1)
                label = labels[predicted.item()]
        st.success(f"Predicted Image is: {label}")
        with st.spinner("Generating text..."):
            set_seed(42)
            result = generator(label, max_length=max_len, num_return_sequences=1)
            st.success("Done!")
            st.write("### Generated Text:")
            st.write(result[0]["generated_text"])
            st.image(image, caption=f"🖼️ Uploaded Image :{label}")

hide_menu = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
