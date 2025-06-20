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

# Predict label from image
def labe(image, model):
    with st.spinner("🔍 Processing Image..."):
        i_tensor = imagepro(image)
        with torch.no_grad():
            outputs = model(i_tensor)
            _, predicted = torch.max(outputs, 1)

    st.session_state["predicted"] = predicted
    st.session_state["topic"] = labels[predicted.item()]

# Show prediction
def predict():
    st.success(f"🎯 Predicted Item: **{st.session_state['topic']}**")

# Generate summary based on topic
def gen(max_len):
    set_seed(42)
    result = generator(st.session_state["topic"], max_length=max_len, num_return_sequences=1)
    st.success("✅ Text Generation Complete!")
    st.write("### 📝 Generated Text:")
    st.write(result[0]["generated_text"])

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🖼️ Image Classification + Text Summary App")

# Upload image
upload = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
max_len = st.slider("✏️ Max Length of Generated Text", 50, 300, 100, step=10)
generate_btn = st.button("🚀 Generate Text")

# If image is uploaded
if upload is not None:
    image = Image.open(upload).convert('RGB')
    image = image.resize((224, 224))
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=False)

    # Predict image
    labe(image, model)
    predict()

    # Generate text if button is clicked
    if generate_btn and "topic" in st.session_state:
        gen(max_len)
