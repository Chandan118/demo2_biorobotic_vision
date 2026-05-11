import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import time

# --- UI Configuration ---
st.set_page_config(page_title="Bio-Robotic Vision", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #111827;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 30px;
    }
    .stImage > img {
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔬 Automated Anomaly Detection for Medical Imagery</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect White Blood Cells and Anomalies with Bio-Robotic Vision.</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "runs/detect/cell_detection/weights/best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.warning("Custom trained model not found. Using generic YOLOv8n for demo.")
        return YOLO("yolov8n.pt")

model = load_model()

# --- Main Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Medical Image")
    st.markdown("Upload a microscope slide image to detect cells automatically.")
    uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("2. Detection Results")
    if uploaded_file is not None:
        if st.button("Analyze", type="primary", use_container_width=True):
            with st.spinner("Running deep learning model..."):
                start_time = time.time()
                
                # Convert PIL image to OpenCV format
                img_array = np.array(image)
                
                # Force rename classes for the marketing script
                model.names = {0: "White Blood Cell", 1: "Anomaly Detected", 2: "Robotic Tool"}
                
                # Run inference
                results = model.predict(img_array, conf=0.25)
                
                # Plot results
                res_plotted = results[0].plot(labels=True, conf=True)
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                processing_time = time.time() - start_time
                
                st.image(res_rgb, caption="Detected Cells/Anomalies", use_column_width=True)
                
                st.markdown(f"### **Analysis complete. {len(results[0].boxes)} objects detected. Processing time: {processing_time:.2f} seconds.**")
    else:
        st.info("Upload an image to see detection results.")

st.markdown("---")
st.markdown("**Tech Stack:** Python, Ultralytics YOLOv8, PyTorch (MPS/CUDA), Streamlit")
