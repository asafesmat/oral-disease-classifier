from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).parent / "resnet50_clean.keras"
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
HIGH_CONFIDENCE = 0.85

st.set_page_config(page_title="Oral Disease Classifier", page_icon="🦷", layout="wide")

st.markdown(
    """
<style>
.hero {
    padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%); color: white;
}
.hero h1 { margin: 0; font-size: 2.2rem; color: white; }
.hero p  { margin: .5rem 0 0; opacity: .92; font-size: 1.05rem; }
.card {
    padding: 1.3rem 1.5rem; border-radius: 16px;
    border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.07);
}
.pred-label { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; opacity: .7; }
.pred-name  { font-size: 2.1rem; font-weight: 700; margin: .1rem 0 .4rem; }
.badge {
    display: inline-block; padding: .25rem .8rem; border-radius: 999px;
    font-size: .85rem; font-weight: 600; color: white;
}
.bar-row   { margin: .55rem 0; }
.bar-top   { display: flex; justify-content: space-between; font-size: .9rem; margin-bottom: .2rem; }
.bar-track { height: 10px; border-radius: 999px; background: rgba(128,128,128,.25); overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0f766e, #1d4ed8); }
.chip {
    display: inline-block; padding: .25rem .75rem; margin: .15rem .2rem .15rem 0;
    border-radius: 999px; border: 1px solid rgba(128,128,128,.4); font-size: .85rem;
}
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": tf.keras.applications.resnet50.preprocess_input},
        safe_mode=False,
        compile=False,
    )


def predict(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize((224, 224))
    x = np.array(img, dtype="float32")[None, ...]
    return load_model().predict(x, verbose=0)[0]


def confidence_badge(p: float) -> str:
    if p >= HIGH_CONFIDENCE:
        color, text = "#15803d", "High confidence"
    elif p >= LOW_CONFIDENCE:
        color, text = "#b45309", "Medium confidence"
    else:
        color, text = "#b91c1c", "Low confidence"
    return f'<span class="badge" style="background:{color}">{text}</span>'


def prob_bar(name: str, p: float) -> str:
    return (
        f'<div class="bar-row"><div class="bar-top"><span>{name}</span>'
        f"<span><b>{p:.1%}</b></span></div>"
        f'<div class="bar-track"><div class="bar-fill" style="width:{p * 100:.1f}%"></div></div></div>'
    )


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About")
    st.write(
        "A deep learning classifier for six oral conditions, built with "
        "transfer learning on ResNet50 (TensorFlow/Keras)."
    )
    c1, c2 = st.columns(2)
    c1.metric("Test accuracy", "95.8%")
    c2.metric("Macro F1", "0.917")
    st.caption("Held-out test set of 527 images.")

    st.subheader("Classes")
    st.markdown(
        "".join(f'<span class="chip">{c}</span>' for c in CLASS_NAMES),
        unsafe_allow_html=True,
    )

    st.subheader("Limitations")
    st.caption(
        "Trained on a single public dataset. Some cases are missed (Caries has the "
        "lowest recall). High confidence does not guarantee a correct result."
    )

# ---------------- Main page ----------------
st.markdown(
    """
<div class="hero">
  <h1>🦷 Oral Disease Image Classifier</h1>
  <p>Upload an oral photo and the model predicts one of six conditions.
  Educational demo only, not a medical diagnosis. Consult a dentist.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Place it next to app.py.")
    st.stop()

file = st.file_uploader("Upload an oral image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if file is None:
    st.info("Upload an image to get a prediction.")
    st.stop()

img = Image.open(file)
with st.spinner("Analyzing..."):
    probs = predict(img)

order = probs.argsort()[::-1]
top = int(order[0])

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.image(img, caption="Uploaded image", width=440)

with right:
    st.markdown(
        f"""
<div class="card">
  <div class="pred-label">Prediction</div>
  <div class="pred-name">{CLASS_NAMES[top]}</div>
  {confidence_badge(float(probs[top]))}
  <p style="margin:.8rem 0 0; opacity:.85">{CLASS_INFO[CLASS_NAMES[top]]}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if probs[top] < LOW_CONFIDENCE:
        st.warning("The image may be unclear or outside the classes this model knows. "
                   "Treat this result with caution.")

    st.markdown("#### Probabilities")
    st.markdown("".join(prob_bar(CLASS_NAMES[i], float(probs[i])) for i in order),
                unsafe_allow_html=True)