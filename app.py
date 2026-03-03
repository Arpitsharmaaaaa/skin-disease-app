import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Page config
st.set_page_config(page_title="Skin Disease Classifier", layout="centered")

st.title("🩺 AI Skin Disease Classifier")
st.write("Upload a dermoscopic image to predict the skin condition.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenetv2_skin_model.keras")
    return model

model = load_model()

# Load class mapping
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())
# Full disease names
# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    CLASS_NAMES = {
        "akiec": "Actinic Keratoses",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevus (Mole)",
        "vasc": "Vascular Lesion"
    }
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    full_name = CLASS_NAMES[predicted_class]

    st.subheader("Prediction:")
    st.success(f"{full_name} ({confidence:.2f}% confidence)")

    st.markdown("---")
    st.warning("⚠ This tool is for educational purposes only. Not a medical diagnosis.")