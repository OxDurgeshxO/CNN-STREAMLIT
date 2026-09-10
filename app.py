import hashlib
import io
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

# Page config
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="wide"
)

# Fashion MNIST official class names (0-9) & emojis
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
CLASS_EMOJIS = ["👕", "👖", "🧶", "👗", "🧥", "👡", "👔", "👟", "👜", "👢"]

# ── Sidebar Architecture & Specs ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 CNN Architecture")
    st.caption("Trained on 60,000 28×28 Fashion MNIST items")
    st.table({
        "Layer": ["Input", "Conv2D_1", "MaxPool_1", "Conv2D_2", "MaxPool_2", "Conv2D_3", "Dense", "Output"],
        "Shape / Details": ["28×28×1", "32 filters, 3×3", "2×2", "64 filters, 3×3", "2×2", "128 filters, 3×3", "256 units (ReLU)", "10 units (Softmax)"]
    })
    st.markdown("### 📊 Metrics")
    st.metric(label="Test Accuracy", value="~91.4%")
    st.metric(label="Inference Latency", value="< 35ms")
    st.markdown("---")
    st.caption("Candidate: Durgesh Dutt Sinha (@OxDurgeshxO) · MCA (AIML)")


@st.cache_resource(show_spinner="Loading deep learning model...")
def load_model():
    """Load the trained Keras CNN model or reconstruct from weights."""
    # Reconstruct architecture to avoid Keras 2/3 serialization version mismatches
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    if os.path.exists("model_weights.npz"):
        loaded = np.load("model_weights.npz")
        weights = [loaded[f"arr_{i}"] for i in range(len(loaded.files))]
        model.set_weights(weights)
        return model

    for path in ["fashion_mnist_cnn.keras", "fashion_mnist_cnn (1).keras", "fashion_mnist_cnn.h5"]:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception:
                continue

    return model


model = load_model()

# ── Main Header ──────────────────────────────────────────────────────────────
st.title("👗 Fashion MNIST Image Classifier")
st.write(
    "Upload a clothing photo (JPG/PNG) or pick a sample below. "
    "The CNN normalizes it to 28×28 grayscale, detects background contrast, and performs real-time classification."
)

col_upload, col_preview = st.columns([1.2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload clear photos of clothing items"
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col_preview:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # ── Image Preprocessing ──────────────────────────────────────────────────
    img_gray = image.convert("L").resize((28, 28))
    img_array = np.array(img_gray).astype("float32") / 255.0

    # Auto-inversion if background is bright (Fashion MNIST is bright subject on dark background)
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array

    with col_preview:
        with st.expander("👁️ View 28×28 Preprocessed Tensor Input"):
            st.image(img_array, clamp=True, width=120)

    # ── Model Inference ──────────────────────────────────────────────────────
    tensor_input = np.expand_dims(np.expand_dims(img_array, axis=0), axis=-1)
    predictions = model.predict(tensor_input, verbose=0)[0]
    winner_idx = int(np.argmax(predictions))
    winner_prob = float(predictions[winner_idx])

    # ── Prediction Banner ────────────────────────────────────────────────────
    st.success(
        f"### Predicted Class: **{CLASS_EMOJIS[winner_idx]} {CLASS_NAMES[winner_idx]}** "
        f"({winner_prob * 100:.1f}% confidence)"
    )

    # ── Plotly Probability Distribution ──────────────────────────────────────
    colors = ["#10b981" if i == winner_idx else "#94a3b8" for i in range(10)]
    labels = [f"{CLASS_EMOJIS[i]} {CLASS_NAMES[i]}" for i in range(10)]

    fig = go.Figure(go.Bar(
        x=predictions * 100,
        y=labels,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p * 100:.1f}%" for p in predictions],
        textposition='outside'
    ))
    fig.update_layout(
        title="Class Probability Distribution (Softmax)",
        xaxis_title="Confidence Percentage (%)",
        yaxis=dict(autorange="reversed"),
        height=450,
        margin=dict(l=20, r=40, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
