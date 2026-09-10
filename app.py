import io
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="wide"
)

# ── Class Definitions ────────────────────────────────────────────────────────
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
CLASS_EMOJIS = ["👕", "👖", "🧶", "👗", "🧥", "👡", "👔", "👟", "👜", "👢"]

# ── Sidebar Architecture & Specs ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 CNN Architecture")
    st.caption("Deep CNN trained on 60,000 Fashion-MNIST images")
    st.table({
        "Layer": ["Input", "Conv2D_1", "MaxPool_1", "Conv2D_2", "MaxPool_2", "Flatten", "Dense_1", "Dense_2 (Output)"],
        "Details": ["28×28×1", "32 filters, 3×3 (ReLU)", "2×2 MaxPool", "64 filters, 3×3 (ReLU)", "2×2 MaxPool", "1600 features", "64 units (ReLU)", "10 units (Softmax)"]
    })
    st.markdown("### ⚡ Inference Engine")
    st.success("Native Vectorized Engine · < 5ms Latency · Zero-Dependency")
    st.markdown("### 📊 Metrics")
    st.metric(label="Test Accuracy", value="~91.4%")
    st.metric(label="Inference Latency", value="< 5ms")
    st.markdown("---")
    st.caption("Candidate: Durgesh Dutt Sinha (@OxDurgeshxO) · MCA (AIML)")


@st.cache_resource(show_spinner="Loading model weights...")
def load_weights():
    """Load model weights from model_weights.npz."""
    weights_path = "model_weights.npz"
    if not os.path.exists(weights_path):
        st.error(f"Missing weights file: `{weights_path}`")
        st.stop()
    loaded = np.load(weights_path)
    return {k: loaded[k] for k in loaded.files}


weights = load_weights()


def predict_fashion(img_28x28, w):
    """High-performance vectorized CNN forward pass in pure NumPy.
    
    Architecture:
      Conv2D(32, 3x3) -> MaxPool(2x2) -> Conv2D(64, 3x3) -> MaxPool(2x2) -> Dense(64) -> Dense(10)
    """
    w0, b0 = w['arr_0'], w['arr_1']  # (3, 3, 1, 32), (32,)
    w1, b1 = w['arr_2'], w['arr_3']  # (3, 3, 32, 64), (64,)
    w2, b2 = w['arr_4'], w['arr_5']  # (1600, 64), (64,)
    w3, b3 = w['arr_6'], w['arr_7']  # (64, 10), (10,)

    x = img_28x28[:, :, np.newaxis]  # (28, 28, 1)
    H, W, C = x.shape

    # 1. Conv2D_1: (28, 28, 1) -> (26, 26, 32)
    out_h, out_w = H - 2, W - 2
    sub_shape = (out_h, out_w, 3, 3, C)
    strides = (x.strides[0], x.strides[1], x.strides[0], x.strides[1], x.strides[2])
    cols = np.lib.stride_tricks.as_strided(x, shape=sub_shape, strides=strides)
    conv1 = np.tensordot(cols, w0, axes=([2, 3, 4], [0, 1, 2])) + b0
    conv1 = np.maximum(0, conv1)  # ReLU

    # 2. MaxPool_1: (26, 26, 32) -> (13, 13, 32)
    pool1 = conv1.reshape(13, 2, 13, 2, 32).max(axis=(1, 3))

    # 3. Conv2D_2: (13, 13, 32) -> (11, 11, 64)
    H2, W2, C2 = pool1.shape
    out_h2, out_w2 = H2 - 2, W2 - 2
    sub_shape2 = (out_h2, out_w2, 3, 3, C2)
    strides2 = (pool1.strides[0], pool1.strides[1], pool1.strides[0], pool1.strides[1], pool1.strides[2])
    cols2 = np.lib.stride_tricks.as_strided(pool1, shape=sub_shape2, strides=strides2)
    conv2 = np.tensordot(cols2, w1, axes=([2, 3, 4], [0, 1, 2])) + b1
    conv2 = np.maximum(0, conv2)  # ReLU

    # 4. MaxPool_2: (11, 11, 64) -> (5, 5, 64)
    pool2 = conv2[:10, :10, :].reshape(5, 2, 5, 2, 64).max(axis=(1, 3))

    # 5. Flatten: 5 * 5 * 64 = 1600
    flat = pool2.reshape(-1)

    # 6. Dense_1: (1600,) @ (1600, 64) + (64,) -> (64,)
    dense1 = np.maximum(0, flat @ w2 + b2)

    # 7. Dense_2: (64,) @ (64, 10) + (10,) -> (10,)
    logits = dense1 @ w3 + b3

    # Numerically stable Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return probs


# ── Main Header ──────────────────────────────────────────────────────────────
st.title("👗 Fashion MNIST Image Classifier")
st.write(
    "Upload a clothing photo (JPG/PNG) or pick one of the sample test items. "
    "The CNN normalizes it to 28×28 grayscale, detects background contrast, and performs real-time classification."
)

# ── Sample Selection Quick Buttons ───────────────────────────────────────────
st.markdown("##### ⚡ Quick Test with Repository Samples:")
if "active_sample" not in st.session_state:
    st.session_state["active_sample"] = "sample_images/T-shirt.png" if os.path.exists("sample_images/T-shirt.png") else None

sample_cols = st.columns(7)
sample_map = [
    ("👕 T-shirt", "sample_images/T-shirt.png"),
    ("👖 Trouser", "sample_images/Trouser.png"),
    ("🧶 Pullover", "sample_images/Pullover.png"),
    ("👗 Dress", "sample_images/Dress.png"),
    ("🧥 Coat", "sample_images/Coat.png"),
    ("👜 Bag", "sample_images/Bag.png"),
    ("👢 Ankle boot", "sample_images/Ankle_boot.png")
]

for i, (label, path) in enumerate(sample_map):
    with sample_cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state["active_sample"] = path

# ── File Upload & Image Resolution ───────────────────────────────────────────
col_upload, col_preview = st.columns([1.2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload clear photos of clothing items"
    )
    if st.session_state.get("active_sample") and uploaded_file is None:
        st.caption(f"Currently testing: `{os.path.basename(st.session_state['active_sample'])}`")

active_image = None
if uploaded_file is not None:
    active_image = Image.open(uploaded_file)
elif st.session_state.get("active_sample") and os.path.exists(st.session_state["active_sample"]):
    active_image = Image.open(st.session_state["active_sample"])

if active_image is not None:
    with col_preview:
        st.image(active_image, caption="Active Input Image", use_container_width=True)

    # ── Image Preprocessing ──────────────────────────────────────────────────
    # Handle RGBA/transparency: composite on white canvas before grayscale conversion
    if active_image.mode in ("RGBA", "LA") or (active_image.mode == "P" and "transparency" in active_image.info):
        rgba_img = active_image.convert("RGBA")
        white_bg = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
        clean_img = Image.alpha_composite(white_bg, rgba_img).convert("L")
    else:
        clean_img = active_image.convert("L")

    img_resized = clean_img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).astype("float32") / 255.0

    # Auto-inversion if background is bright (Fashion MNIST is bright subject on dark background)
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array

    with col_preview:
        with st.expander("👁️ View 28×28 Model Tensor Input"):
            st.image(img_array, clamp=True, width=120)

    # ── Model Inference ──────────────────────────────────────────────────────
    predictions = predict_fashion(img_array, weights)
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
