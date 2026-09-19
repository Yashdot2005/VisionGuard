"""
VisionGuard - Interactive Streamlit Quality Control Dashboard
Live Demo Application for Manufacturing Line Automated Defect Inspection.
"""

import os
from pathlib import Path
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd

from dataset import load_and_preprocess_image
from model import build_model

st.set_page_config(
    page_title="VisionGuard | Defect Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card-pass {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-card-fail {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_visionguard_model(checkpoint_path: str):
    """Load model once and cache in memory."""
    if not os.path.exists(checkpoint_path):
        return None, None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", "resnet18")
    class_names = checkpoint.get("class_names", ["defective", "normal"])
    image_size = checkpoint.get("image_size", 224)

    model = build_model(architecture=architecture, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names, device, image_size


def main():
    st.markdown('<div class="main-header">🛡️ VisionGuard: Manufacturing Defect Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Computer Vision Quality Control System for Industrial Production Lines</div>', unsafe_allow_html=True)

    # Sidebar settings
    st.sidebar.header("⚙️ System Configuration")
    checkpoint_path = st.sidebar.text_input("Model Checkpoint", value="checkpoints/best_model.pt")

    model, class_names, device, image_size = load_visionguard_model(checkpoint_path)

    if model is None:
        st.error(f"Checkpoint not found at `{checkpoint_path}`. Please run `train.py` first to train a model!")
        return

    st.sidebar.success(f"Model Loaded: **ResNet-18** (Device: `{device}`)")

    # Navigation tabs
    tab_inspect, tab_metrics, tab_about = st.tabs(["🔍 Live Inspection", "📊 Performance Analytics", "ℹ️ About VisionGuard"])

    with tab_inspect:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. Select or Upload Image")
            input_mode = st.radio("Input Source:", ["Sample Images", "Upload Image"], horizontal=True)

            selected_image = None
            if input_mode == "Sample Images":
                sample_files = {
                    "Normal Machined Surface": "sample_images/test_sample_normal.jpg",
                    "Defective Component": "sample_images/test_sample_defective.jpg"
                }
                choice = st.selectbox("Choose Sample:", list(sample_files.keys()))
                sample_path = sample_files[choice]
                if os.path.exists(sample_path):
                    selected_image = Image.open(sample_path)
                    st.image(selected_image, caption=f"Selected: {choice}", use_container_width=True)
                else:
                    st.warning("Sample images not found. Run `prepare_data.py` first.")
            else:
                uploaded = st.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg", "bmp"])
                if uploaded:
                    selected_image = Image.open(uploaded)
                    st.image(selected_image, caption="Uploaded Product Image", use_container_width=True)

        with col2:
            st.subheader("2. Automated Inspection Verdict")
            if selected_image is not None:
                # Save temp image for processing
                temp_path = "temp_inspect.png"
                selected_image.save(temp_path)

                tensor = load_and_preprocess_image(temp_path, image_size=image_size).to(device)
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

                defect_prob = probs[0] if class_names[0] == "defective" else probs[1]
                normal_prob = probs[1] if class_names[0] == "defective" else probs[0]

                is_defect = defect_prob > 0.5
                confidence = max(defect_prob, normal_prob) * 100.0

                if is_defect:
                    st.markdown(f"""
                    <div class="status-card-fail">
                        <div class="status-title" style="color: #DC2626;">❌ DEFECTIVE [REJECT]</div>
                        <p style="margin-top: 6px; color: #7F1D1D; font-size: 1.1rem;">Surface anomaly detected on component line.</p>
                        <strong style="font-size: 1.2rem; color: #991B1B;">Confidence: {confidence:.2f}%</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-card-pass">
                        <div class="status-title" style="color: #059669;">✅ NORMAL [APPROVED]</div>
                        <p style="margin-top: 6px; color: #065F46; font-size: 1.1rem;">Component passed all industrial quality standards.</p>
                        <strong style="font-size: 1.2rem; color: #047857;">Confidence: {confidence:.2f}%</strong>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("#### Confidence Breakdown:")
                prob_data = pd.DataFrame({
                    "Class": ["Defective", "Normal"],
                    "Probability": [defect_prob, normal_prob]
                })
                st.bar_chart(prob_data.set_index("Class"), height=220)

                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                st.info("Select or upload an image on the left to trigger the inspection.")

    with tab_metrics:
        st.subheader("Model Evaluation & Training Metrics")
        m_col1, m_col2 = st.columns([1, 1])

        with m_col1:
            if os.path.exists("reports/confusion_matrix.png"):
                st.image("reports/confusion_matrix.png", caption="Confusion Matrix on Test Split", use_container_width=True)
            if os.path.exists("reports/training_curves.png"):
                st.image("reports/training_curves.png", caption="Training & Validation Curves", use_container_width=True)

        with m_col2:
            if os.path.exists("reports/roc_curve.png"):
                st.image("reports/roc_curve.png", caption="Receiver Operating Characteristic (ROC-AUC)", use_container_width=True)
            if os.path.exists("reports/performance_report.md"):
                with open("reports/performance_report.md", "r") as f:
                    st.markdown(f.read())

    with tab_about:
        st.subheader("VisionGuard Specifications")
        st.markdown("""
        - **Organization:** Code A Nova Machine Learning Internship
        - **Target Domain:** Automated Manufacturing Quality Assurance
        - **Architecture:** ResNet-18 Deep Residual Network (Fine-tuned last two residual blocks)
        - **Optimization:** Adam Optimizer with Learning Rate Scheduling and Early Stopping
        - **Tracking:** TensorBoard Scalar Logging and Scikit-learn Metric Reporting
        """)


if __name__ == "__main__":
    main()
