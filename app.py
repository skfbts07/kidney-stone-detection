import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load model and tools
model = joblib.load("xgb_model.pkl")
pca = joblib.load("pca_transform.pkl")
class_names = joblib.load("class_labels.pkl")

# App title
st.title("🩺 Kidney Stone Detector")
st.write("Upload a kidney scan image to check for kidney stones.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    arr = np.array(image) / 255.0
    flat = arr.reshape(1, -1)
    reduced = pca.transform(flat)
    prediction = model.predict(reduced)[0]
    label = class_names[prediction]

    # Show result
    st.success(f"🔍 Prediction: **{label}**")