import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

st.set_page_config(
    page_title="Disaster Image Classifier",
    layout="centered"
)

st.title("Disaster Image Classification App")
st.write("Upload a disaster image to classify what disaster it is")

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    
    in_features = model.classifier[1].in_features
    NUM_CLASSES = 4
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=in_features, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=NUM_CLASSES)
    )
    
    model.load_state_dict(torch.load('custom-mobilenet-model.pth', map_location=torch.device('cpu')))
    
    model.eval()
    return model

CLASS_LABELS = ['Earthquake', 'Land Slide', 'Urban Fire', 'Water Disaster']

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    return img_tensor

def predict(model, processed_image):
    """
    Make prediction using the PyTorch model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    processed_image = processed_image.to(device)
    
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    return probabilities.cpu().numpy()

st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Classification Results")
        if st.button("Classify Image", type="primary"):
            with st.spinner("Classifying..."):
                try:
                    model = load_model()
                    processed_image = preprocess_image(image)
                    predictions = predict(model, processed_image)
                    
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx]
                    
                    if confidence >= confidence_threshold:
                        st.success(f"**Predicted Class:** {CLASS_LABELS[predicted_class_idx]}")
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    else:
                        st.warning(f"Low confidence prediction: {CLASS_LABELS[predicted_class_idx]} ({confidence*100:.2f}%)")
                    
                    st.subheader("All Class Probabilities")
                    for i, (label, prob) in enumerate(zip(CLASS_LABELS, predictions[0])):
                        st.progress(float(prob), text=f"{label}: {prob*100:.2f}%")
                
                except Exception as e:

                    st.error(f"Error: {e}")

