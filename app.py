"""Streamlit app: Bird Species Classification using a pretrained EfficientNet-B2 model.

Model: dennisjooo/Birds-Classifier-EfficientNetB2 (Hugging Face Hub).
It is downloaded automatically on the first run (internet needed) and cached on your computer.
"""
import streamlit as st
from PIL import Image, ImageOps
from transformers import pipeline

MODEL_ID = "dennisjooo/Birds-Classifier-EfficientNetB2"

st.set_page_config(page_title="Bird Species Classifier", page_icon="🐦")


@st.cache_resource(show_spinner="Loading model (first run downloads it)...")
def load_classifier():
    return pipeline("image-classification", model=MODEL_ID)


st.title("🐦 Bird Species Classification")
st.write("Upload a bird image and the EfficientNet-B2 model will predict its species.")

try:
    classifier = load_classifier()
except Exception as error:
    st.error(f"Could not load the model. Check your internet connection and installed packages. Details: {error}")
    st.stop()

uploaded_file = st.file_uploader("Choose a bird image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        # 1. Read and display the image
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
        st.image(image, caption="Uploaded image")

        # 2. Predict (the pipeline resizes and normalizes the image automatically)
        with st.spinner("Identifying the bird species..."):
            results = classifier(image, top_k=5)

        # 3. Show the best prediction and the top 5
        best = results[0]
        st.subheader("Prediction")
        st.success(best["label"].title())
        st.metric("Confidence", f"{best['score'] * 100:.2f}%")

        st.subheader("Top 5 predictions")
        for item in results:
            st.write(f"**{item['label'].title()}** — {item['score'] * 100:.2f}%")
            st.progress(min(max(float(item["score"]), 0.0), 1.0))
    except Exception as error:
        st.error(f"Could not process this image: {error}")
else:
    st.info("Upload a JPG, PNG or WEBP image to begin.")
