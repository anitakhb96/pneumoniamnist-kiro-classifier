"""
app.py — Streamlit inference UI for the PneumoniaMNIST Classifier.

Two input modes (sidebar radio):
  1. "Select test example"  — sample a random image from the test set
  2. "Upload image"         — upload a PNG/JPG, converted to grayscale 28×28

Displays:
  - Input image (upscaled for visibility)
  - Predicted class: Normal / Pneumonia
  - Predicted probability for the Pneumonia class (raw sigmoid output)
  - Disclaimer: for educational/research use only

Run:
    streamlit run app.py

IMPORTANT: Requires outputs/best_model.pth and outputs/norm_stats.json.
           Run train.py first if these files are missing.

Educational/research use only. Not for clinical diagnosis.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# --- Resolve project root regardless of working directory ---
PROJECT_ROOT = Path(__file__).parent.resolve()

import sys
sys.path.insert(0, str(PROJECT_ROOT))

import config
from model import get_model

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PneumoniaMNIST Classifier",
    page_icon="🫁",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained model from disk. Cached across reruns."""
    if not config.MODEL_PATH.exists():
        return None
    model = get_model(seed=config.SEED, device=torch.device("cpu"))
    state = torch.load(config.MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading normalisation stats...")
def load_norm_stats():
    """Load training-set normalisation stats from disk."""
    if not config.NORM_STATS_PATH.exists():
        return None, None
    with open(config.NORM_STATS_PATH, "r") as f:
        stats = json.load(f)
    return float(stats["mean"]), float(stats["std"])


@st.cache_data(show_spinner="Loading test set...")
def load_test_images():
    """
    Load the raw test set images and labels for the 'Select test example' mode.
    Images are returned as numpy arrays (H, W) in [0, 1].
    """
    import medmnist
    from medmnist import PneumoniaMNIST
    dataset = PneumoniaMNIST(split="test", transform=transforms.ToTensor(), download=True)
    images, labels = [], []
    for img, lbl in dataset:
        images.append(img.squeeze().numpy())     # (28, 28) float32
        labels.append(int(lbl.item()))
    return images, labels


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict(model, image_pil: Image.Image, mean: float, std: float) -> tuple:
    """
    Run inference on a PIL Image.

    Returns
    -------
    pred_class : str    "Normal" or "Pneumonia"
    prob_pneumonia : float   raw sigmoid probability for the Pneumonia class
    """
    # Resize to 28×28, convert to grayscale L
    img = image_pil.convert("L").resize((28, 28), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])
    tensor = transform(img).unsqueeze(0)    # (1, 1, 28, 28)

    with torch.no_grad():
        logit = model(tensor)               # (1, 1)
        prob = torch.sigmoid(logit).item()  # scalar in [0, 1]

    pred_class = config.CLASS_NAMES[int(prob >= 0.5)]
    return pred_class, prob


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def main():
    st.title("🫁 PneumoniaMNIST Diagnostic Classifier")
    st.caption(
        "A lightweight CNN trained on the PneumoniaMNIST dataset to classify "
        "chest X-ray images as Normal or Pneumonia."
    )

    # --- Load model and stats ---
    model = load_model()
    mean, std = load_norm_stats()

    if model is None:
        st.error(
            "⚠️ Model checkpoint not found at `outputs/best_model.pth`.\n\n"
            "Please run `python train.py` first to train and save the model."
        )
        st.stop()

    if mean is None:
        st.error(
            "⚠️ Normalisation statistics not found at `outputs/norm_stats.json`.\n\n"
            "Please run `python train.py` first."
        )
        st.stop()

    # --- Sidebar: input mode ---
    st.sidebar.header("Input Mode")
    mode = st.sidebar.radio(
        "Choose input source:",
        options=["Select test example", "Upload image"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**ℹ️ About this model**\n\n"
        f"- Architecture: PneumoniaCNN\n"
        f"- Parameters: 105,329\n"
        f"- Training epochs: {config.NUM_EPOCHS}\n"
        f"- Input: 28×28 grayscale\n"
    )

    # ------------------------------------------------------------------
    # Mode 1: Select test example
    # ------------------------------------------------------------------
    if mode == "Select test example":
        st.subheader("Test Set Example")

        col_btn, col_spacer = st.columns([1, 3])
        with col_btn:
            new_sample = st.button("🔀 Sample new image")

        # Store selected index in session state so it persists across reruns
        if "test_idx" not in st.session_state or new_sample:
            test_images, test_labels = load_test_images()
            st.session_state["test_idx"] = random.randint(0, len(test_images) - 1)
            st.session_state["test_images"] = test_images
            st.session_state["test_labels"] = test_labels

        idx = st.session_state["test_idx"]
        test_images = st.session_state["test_images"]
        test_labels = st.session_state["test_labels"]

        img_array = test_images[idx]                          # (28, 28) float32
        true_label = config.CLASS_NAMES[test_labels[idx]]

        # Convert to PIL for display and inference (uint8 2D array → grayscale L)
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))

        # Run prediction
        pred_class, prob_pneumonia = predict(model, img_pil, mean, std)

        # Display
        col_img, col_results = st.columns([1, 2])

        with col_img:
            # Upscale 28×28 → 168×168 for visibility
            img_display = img_pil.resize((168, 168), Image.NEAREST)
            st.image(img_display, caption=f"Test image #{idx}", use_container_width=False)

        with col_results:
            st.markdown(f"**True label:** `{true_label}`")
            st.markdown("---")
            st.markdown(f"**Predicted class:** `{pred_class}`")
            st.markdown(f"**Pneumonia probability:** `{prob_pneumonia:.4f}`")

            # Visual probability bar
            st.progress(prob_pneumonia)

            # Match indicator
            if pred_class == true_label:
                st.success("✅ Prediction matches true label")
            else:
                st.warning("⚠️ Prediction does not match true label")

    # ------------------------------------------------------------------
    # Mode 2: Upload image
    # ------------------------------------------------------------------
    else:
        st.subheader("Upload a Chest X-ray Image")
        uploaded_file = st.file_uploader(
            "Upload a PNG or JPG image (will be resized to 28×28 grayscale):",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            img_pil = Image.open(uploaded_file)

            # Run prediction
            pred_class, prob_pneumonia = predict(model, img_pil, mean, std)

            col_img, col_results = st.columns([1, 2])

            with col_img:
                # Show original upload, then resized version
                st.image(img_pil, caption="Uploaded image (original)", use_container_width=True)
                resized = img_pil.convert("L").resize((28, 28), Image.LANCZOS)
                resized_display = resized.resize((112, 112), Image.NEAREST)
                st.image(resized_display, caption="Input to model (28×28 grayscale)")

            with col_results:
                st.markdown(f"**Predicted class:** `{pred_class}`")
                st.markdown(f"**Pneumonia probability:** `{prob_pneumonia:.4f}`")

                # Visual probability bar
                st.progress(prob_pneumonia)
        else:
            st.info("👆 Upload an image above to run the classifier.")

    # ------------------------------------------------------------------
    # Disclaimer
    # ------------------------------------------------------------------
    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** This tool is for **educational and research demonstration only**. "
        "It must **not** be used for clinical diagnosis or any medical decision-making. "
        "Always consult a qualified healthcare professional for medical advice."
    )


if __name__ == "__main__":
    main()
