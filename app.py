from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).parent / "best_oral_disease_model.keras"
CLASS_NAMES = ["Calculus", "Caries", "Discoloration",
               "Gingivitis", "Mouth Ulcer", "hypodontia"]
CLASS_INFO = {
    "Calculus": "Hardened dental plaque (tartar) on the teeth.",
    "Caries": "Tooth decay caused by acid-producing bacteria.",
    "Discoloration": "Staining or color change of the teeth.",
    "Gingivitis": "Inflammation of the gums.",
    "Mouth Ulcer": "Open sore on the inner lining of the mouth.",
    "hypodontia": "Congenitally missing teeth.",
}
LOW_CONFIDENCE = 0.60

# Test-set results (527 images) from the training notebook
TEST_REPORT = pd.DataFrame(
    {
        "Precision": [0.940, 0.952, 0.815, 0.973, 0.947, 0.973],
        "Recall":    [0.940, 0.690, 0.846, 0.978, 1.000, 0.995],
        "F1-score":  [0.940, 0.800, 0.830, 0.976, 0.973, 0.984],
        "Images":    [67, 29, 26, 186, 36, 183],
    },
    index=CLASS_NAMES,
)

st.set_page_config(page_title="Oral Disease Classifier", page_icon="🦷", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": tf.keras.applications.resnet50.preprocess_input},
        safe_mode=False,
    )


def predict(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize((224, 224))
    x = np.array(img, dtype="float32")[None, ...]
    return load_model().predict(x, verbose=0)[0]


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About this project")
    st.write(
        "A deep learning classifier for six oral conditions, built with "
        "transfer learning on ResNet50 (TensorFlow/Keras)."
    )
    c1, c2 = st.columns(2)
    c1.metric("Test accuracy", "95.8%")
    c2.metric("Macro F1", "0.917")
    st.caption("Evaluated on a held-out test set of 527 images.")

    st.subheader("Per-class results")
    st.dataframe(TEST_REPORT.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}",
                                           "F1-score": "{:.2f}"}))

    st.subheader("Limitations")
    st.markdown(
        "- Trained on a single public dataset.\n"
        "- Caries has the lowest recall (0.69): some cases are missed.\n"
        "- High confidence does not guarantee a correct result."
    )

# ---------------- Main page ----------------
st.title("🦷 Oral Disease Image Classifier")
st.caption("Educational demo only. This is not a medical diagnosis; consult a dentist.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Place it next to app.py.")
    st.stop()

file = st.file_uploader("Upload an oral image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if file is None:
    st.info("Upload an image to get a prediction.")
    st.stop()

img = Image.open(file)
probs = predict(img)
order = probs.argsort()[::-1]
top = int(order[0])

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.image(img, caption="Uploaded image", use_container_width=True)

with right:
    st.subheader(f"Prediction: {CLASS_NAMES[top]}")
    st.metric("Confidence", f"{probs[top]:.1%}")
    st.write(CLASS_INFO[CLASS_NAMES[top]])

    if probs[top] < LOW_CONFIDENCE:
        st.warning("Low confidence. The image may be unclear or outside the classes "
                   "this model knows. Treat this result with caution.")

    st.markdown("**Top 3 predictions**")
    for i in order[:3]:
        st.progress(float(probs[i]), text=f"{CLASS_NAMES[i]}: {probs[i]:.1%}")

    with st.expander("All class probabilities"):
        st.dataframe(
            pd.DataFrame({"Probability": probs}, index=CLASS_NAMES)
            .sort_values("Probability", ascending=False)
            .style.format("{:.1%}")
        )