import streamlit as st
import requests
import torch
from PIL import Image   
import torchvision.transforms as transforms
import torchvision.models as models
import streamlit as st
from transformers import pipeline, set_seed
import cv2

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

generator = load_generator()


st.title("Image Classification App And Provide Summary")

model = models.resnet18(pretrained=True)
model.eval()

LABELS_URL ="https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
label = [line.strip() for line in requests.get(LABELS_URL).text.splitlines()]

def imagepro(image):
    transform = transforms.Compose([
        transforms.Resize((224,244)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)
def labe(image):
    with st.spinner("Processing..."):
        i_tensor = imagepro(image)
        with torch.no_grad():
            outputs = model(i_tensor)
            a, predicted = torch.max(outputs, 1)
            label = label[predicted.item()]
    
    st.success(f"Predicted Items is: {label}")
    topic = label
    return topic
def gen(topic,max_len,):
    set_seed(42)
    result = generator(label, max_length=max_len, num_return_sequences=1)
    st.success("Done!")
    st.write("### Generated Text:")
    st.write(result[0]["generated_text"])
    
gun=st.button("Generate text" )
upload = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
max_len = st.slider("Max length Of Text", 50, 300, 100, step=10)

if upload is not None:
    image = Image.open(upload).convert('RGB')
    image=image.resize((224,224))
    if gun:
        gen(label,max_label)
   
    
    st.image(image, caption="Uploaded Image")
     labe(image)

   
    max_len = st.slider("Max length", 50, 300, 100, step=10)

    if st.button("Generate"):
        with st.spinner("Generating text..."):
            set_seed(42)
            result = generator(label, max_length=max_len, num_return_sequences=1)
            st.success("Done!")
            st.write("### Generated Text:")
            st.write(result[0]["generated_text"])
