"""
app.py — Streamlit inference UI for the PneumoniaMNIST Classifier.

Supports two models via a sidebar selector:
  • 28×28 Baseline       — original two-block CNN, threshold = 0.5
  • 64×64 Enhanced Pipeline — three-block CNN, shows predictions at both
                               threshold = 0.5 and the Youden-optimal τ*

Two input modes (sidebar radio):
  1. "Select test example"  — sample a random image from the test set of the
                               selected model's resolution
  2. "Upload image"         — upload a PNG/JPG, converted to grayscale and
                               resized to the model's native resolution

All original 28×28 behaviour is preserved when the baseline model is selected.

Run:
    streamlit run app.py

IMPORTANT: Requires the relevant model checkpoint and norm_stats.json.
  28×28 model : outputs/best_model.pth  and  outputs/norm_stats.json
  64×64 model : outputs_64/best_model.pth  and  outputs_64/norm_stats.json
                (also outputs_64/threshold.json for the optimal threshold)
Run the appropriate train / evaluate scripts if files are missing.

Educational/research use only. Not for clinical diagnosis.
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# --- Resolve project root regardless of working directory ---
PROJECT_ROOT = Path(__file__).parent.resolve()

import sys
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PneumoniaMNIST Classifier",
    page_icon="🫁",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Model configuration registry
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Callable

@dataclass
class ModelConfig:
    key:              str
    display_name:     str
    image_size:       int
    model_path:       Path
    norm_stats_path:  Path
    threshold_path:   Optional[Path]   # None for 28×28 (uses fixed 0.5)
    metrics_path:     Path
    param_count:      int
    arch_name:        str


def _make_configs() -> dict:
    import config as cfg28
    import config_64 as cfg64

    return {
        "28x28": ModelConfig(
            key             = "28x28",
            display_name    = "28×28 Baseline",
            image_size      = 28,
            model_path      = cfg28.MODEL_PATH,
            norm_stats_path = cfg28.NORM_STATS_PATH,
            threshold_path  = None,
            metrics_path    = cfg28.METRICS_PATH,
            param_count     = 105_329,
            arch_name       = "PneumoniaCNN (2 blocks)",
        ),
        "64x64": ModelConfig(
            key             = "64x64",
            display_name    = "64×64 Enhanced Pipeline",
            image_size      = 64,
            model_path      = cfg64.MODEL_PATH,
            norm_stats_path = cfg64.NORM_STATS_PATH,
            threshold_path  = cfg64.THRESHOLD_PATH,
            metrics_path    = cfg64.METRICS_PATH,
            param_count     = 140_000,   # approximate; updated by info card
            arch_name       = "PneumoniaCNN64 (3 blocks + AdaptiveAvgPool)",
        ),
    }


# ---------------------------------------------------------------------------
# Cached resource loaders (keyed by model key so both can coexist in cache)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model(model_key: str):
    """Load and cache the trained model for a given model key."""
    configs = _make_configs()
    cfg     = configs[model_key]

    if not cfg.model_path.exists():
        return None

    if model_key == "28x28":
        from model import PneumoniaCNN
        import config as cfg28
        torch.manual_seed(cfg28.SEED)
        model = PneumoniaCNN()
    else:
        from model_64 import PneumoniaCNN64
        import config_64 as cfg64
        torch.manual_seed(cfg64.SEED)
        model = PneumoniaCNN64()

    state = torch.load(cfg.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading normalisation stats...")
def load_norm_stats(model_key: str):
    """Load and cache normalisation stats for a given model key."""
    configs = _make_configs()
    cfg     = configs[model_key]

    if not cfg.norm_stats_path.exists():
        return None, None

    with open(cfg.norm_stats_path, "r") as f:
        stats = json.load(f)
    return float(stats["mean"]), float(stats["std"])


@st.cache_data(show_spinner="Loading test set...")
def load_test_images(model_key: str):
    """
    Load raw test-set images for the selected resolution.
    Images are returned as numpy arrays (H, W) in [0, 1].
    """
    from medmnist import PneumoniaMNIST

    size_kwarg = {} if model_key == "28x28" else {"size": 64}
    dataset = PneumoniaMNIST(
        split="test",
        transform=transforms.ToTensor(),
        download=True,
        **size_kwarg,
    )
    images, labels = [], []
    for img, lbl in dataset:
        images.append(img.squeeze().numpy())
        labels.append(int(lbl.item()))
    return images, labels


def load_threshold(threshold_path: Optional[Path]) -> Optional[float]:
    """Load τ* from threshold.json, or return None if not available."""
    if threshold_path is None or not threshold_path.exists():
        return None
    with open(threshold_path, "r") as f:
        d = json.load(f)
    return float(d.get("tau_star", 0.5))


def load_auroc(metrics_path: Path, model_key: str) -> Optional[float]:
    """Read the AUROC from a metrics.json for the sidebar info card."""
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r") as f:
            m = json.load(f)
        if model_key == "64x64":
            return float(m["threshold_05"]["auroc"])
        return float(m.get("auroc", 0.0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict(
    model,
    image_pil:  Image.Image,
    mean:       float,
    std:        float,
    image_size: int,
) -> tuple:
    """
    Run inference on a PIL Image.

    Returns
    -------
    prob_pneumonia : float  — raw sigmoid probability in [0, 1]
    """
    img = image_pil.convert("L").resize((image_size, image_size), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])
    tensor = transform(img).unsqueeze(0)   # (1, 1, H, W)

    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()

    return float(prob)


def classify(prob: float, threshold: float, class_names: list) -> str:
    return class_names[int(prob >= threshold)]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(configs: dict) -> tuple:
    st.sidebar.header("Model")
    model_choice = st.sidebar.radio(
        "Select model:",
        options=list(configs.keys()),
        format_func=lambda k: configs[k].display_name,
        index=0,
    )

    cfg = configs[model_choice]

    # Model info card
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ℹ️ Model Info**")
    st.sidebar.markdown(f"- Architecture: {cfg.arch_name}")
    st.sidebar.markdown(f"- Input: {cfg.image_size}×{cfg.image_size} grayscale")

    auroc = load_auroc(cfg.metrics_path, model_choice)
    if auroc is not None:
        st.sidebar.markdown(f"- Test AUROC: `{auroc:.4f}`")
    else:
        st.sidebar.markdown("- Test AUROC: *not yet evaluated*")

    tau_star = load_threshold(cfg.threshold_path)
    if tau_star is not None:
        st.sidebar.markdown(f"- Optimal threshold (τ*): `{tau_star:.4f}` *(Youden J, val set)*")

    st.sidebar.markdown("---")

    st.sidebar.header("Input Mode")
    input_mode = st.sidebar.radio(
        "Choose input source:",
        options=["Select test example", "Upload image"],
        index=0,
        key=f"input_mode_{model_choice}",
    )

    return model_choice, input_mode


# ---------------------------------------------------------------------------
# Mode renderers
# ---------------------------------------------------------------------------

def render_test_example(
    model,
    cfg:        ModelConfig,
    model_key:  str,
    mean:       float,
    std:        float,
    tau_star:   Optional[float],
) -> None:
    st.subheader(f"Test Set Example — {cfg.display_name}")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        new_sample = st.button("🔀 Sample new image")

    session_key_idx    = f"test_idx_{model_key}"
    session_key_images = f"test_images_{model_key}"
    session_key_labels = f"test_labels_{model_key}"

    if session_key_idx not in st.session_state or new_sample:
        test_images, test_labels = load_test_images(model_key)
        st.session_state[session_key_idx]    = random.randint(0, len(test_images) - 1)
        st.session_state[session_key_images] = test_images
        st.session_state[session_key_labels] = test_labels

    idx         = st.session_state[session_key_idx]
    test_images = st.session_state[session_key_images]
    test_labels = st.session_state[session_key_labels]

    img_array  = test_images[idx]
    true_label = ["Normal", "Pneumonia"][test_labels[idx]]
    img_pil    = Image.fromarray((img_array * 255).astype(np.uint8))

    prob = predict(model, img_pil, mean, std, cfg.image_size)

    col_img, col_results = st.columns([1, 2])

    with col_img:
        img_display = img_pil.resize((168, 168), Image.NEAREST)
        st.image(img_display, caption=f"Test image #{idx} ({cfg.image_size}×{cfg.image_size})")

    with col_results:
        st.markdown(f"**True label:** `{true_label}`")
        st.markdown("---")

        pred_05 = classify(prob, 0.5, ["Normal", "Pneumonia"])
        st.markdown(f"**Predicted (threshold = 0.5):** `{pred_05}`")
        st.markdown(f"**Pneumonia probability:** `{prob:.4f}`")
        st.progress(prob)

        if tau_star is not None:
            pred_opt = classify(prob, tau_star, ["Normal", "Pneumonia"])
            st.markdown(f"**Predicted (τ* = {tau_star:.4f}, Youden J):** `{pred_opt}`")

        st.markdown("---")
        match_pred = pred_05
        if match_pred == true_label:
            st.success("✅ Prediction (0.5) matches true label")
        else:
            st.warning("⚠️ Prediction (0.5) does not match true label")


def render_upload(
    model,
    cfg:      ModelConfig,
    mean:     float,
    std:      float,
    tau_star: Optional[float],
) -> None:
    st.subheader(f"Upload a Chest X-ray Image — {cfg.display_name}")
    uploaded_file = st.file_uploader(
        f"Upload a PNG or JPG (resized to {cfg.image_size}×{cfg.image_size} grayscale):",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file)
        prob    = predict(model, img_pil, mean, std, cfg.image_size)

        col_img, col_results = st.columns([1, 2])

        with col_img:
            st.image(img_pil, caption="Uploaded image (original)", use_container_width=True)
            resized       = img_pil.convert("L").resize((cfg.image_size, cfg.image_size), Image.LANCZOS)
            resized_disp  = resized.resize((112, 112), Image.NEAREST)
            st.image(resized_disp, caption=f"Model input ({cfg.image_size}×{cfg.image_size} grayscale)")

        with col_results:
            pred_05 = classify(prob, 0.5, ["Normal", "Pneumonia"])
            st.markdown(f"**Predicted (threshold = 0.5):** `{pred_05}`")
            st.markdown(f"**Pneumonia probability:** `{prob:.4f}`")
            st.progress(prob)

            if tau_star is not None:
                pred_opt = classify(prob, tau_star, ["Normal", "Pneumonia"])
                st.markdown(f"**Predicted (τ* = {tau_star:.4f}, Youden J):** `{pred_opt}`")
    else:
        st.info("👆 Upload an image above to run the classifier.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.title("🫁 PneumoniaMNIST Diagnostic Classifier")
    st.caption(
        "A lightweight CNN trained on PneumoniaMNIST to classify chest X-ray images "
        "as Normal or Pneumonia.  Choose a model in the sidebar."
    )

    configs = _make_configs()
    model_key, input_mode = render_sidebar(configs)
    cfg = configs[model_key]

    # --- Load model and stats ---
    model = load_model(model_key)
    mean, std = load_norm_stats(model_key)

    if model is None:
        st.error(
            f"⚠️ Model checkpoint not found at `{cfg.model_path}`.\n\n"
            f"Please run `python train{'_64' if model_key == '64x64' else ''}.py` first."
        )
        st.stop()

    if mean is None:
        st.error(
            f"⚠️ Normalisation statistics not found at `{cfg.norm_stats_path}`.\n\n"
            f"Please run `python train{'_64' if model_key == '64x64' else ''}.py` first."
        )
        st.stop()

    tau_star = load_threshold(cfg.threshold_path)
    if model_key == "64x64" and tau_star is None and cfg.threshold_path is not None:
        st.warning(
            f"⚠️ Optimal threshold not found at `{cfg.threshold_path}`.  "
            "Only the 0.5 threshold will be used.  "
            "Run `python evaluate_64.py` to compute τ*."
        )

    # --- Render selected input mode ---
    if input_mode == "Select test example":
        render_test_example(model, cfg, model_key, mean, std, tau_star)
    else:
        render_upload(model, cfg, mean, std, tau_star)

    # --- Disclaimer ---
    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** This tool is for **educational and research demonstration only**. "
        "It must **not** be used for clinical diagnosis or any medical decision-making. "
        "Always consult a qualified healthcare professional for medical advice."
    )


if __name__ == "__main__":
    main()
