import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# -----------------------
# App Config
# -----------------------
st.set_page_config(page_title="Facial Expression Recognition", layout="centered")
st.title("😄 Facial Expression Recognition")
st.write("Upload a face image and the model will predict the emotion.")

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_cnn_model():
    return load_model("fer2013_cnn_model.h5")  
    # OR if SavedModel:
    # return load_model("models/fer2013_cnn_model")

model = load_cnn_model()

# Emotion labels (FER-2013 order)
emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# -----------------------
# Image Preprocessing
# -----------------------
def preprocess_image(image):
    image = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48))

    # Normalize
    normalized = resized / 255.0

    # Reshape for model
    reshaped = normalized.reshape(1, 48, 48, 1)

    return reshaped

# -----------------------
# File Upload
# -----------------------
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    # Predict
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Results
    st.subheader("🧠 Prediction")
    st.write(f"**Emotion:** {emotion_labels[predicted_class]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Probability breakdown
    st.subheader("📊 Class Probabilities")
    for i, emotion in enumerate(emotion_labels):
        st.write(f"{emotion}: {predictions[0][i]*100:.2f}%")
