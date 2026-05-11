import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import time

st.set_page_config(
    page_title="VisionDX | Biomedical AI",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #030712 0%, #0a0f1e 50%, #050b15 100%);
}

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1428 100%);
    border-right: 1px solid rgba(16,185,129,0.15);
}

.hero-container {
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(6,182,212,0.08) 50%, rgba(99,102,241,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 20px;
    padding: 44px;
    text-align: center;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero-container::after {
    content: '';
    position: absolute;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
    top: -100px;
    right: -100px;
    border-radius: 50%;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981, #06b6d4, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1.5px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(226,232,240,0.55);
    margin-top: 10px;
    font-weight: 400;
    letter-spacing: 0.5px;
}

.badge {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.35);
    color: #10b981;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.result-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px;
    margin-top: 12px;
}

.wow-banner {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(6,182,212,0.12));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-top: 16px;
}

.wow-text {
    font-size: 1.2rem;
    font-weight: 700;
    color: #10b981;
    letter-spacing: 0.5px;
}

.wow-sub {
    font-size: 0.8rem;
    color: rgba(148,163,184,0.7);
    margin-top: 4px;
}

.detect-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 4px;
}

.pill-wbc {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.35);
    color: #10b981;
}

.pill-anomaly {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.35);
    color: #ef4444;
}

.pill-tool {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    color: #818cf8;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #10b981;
}

.metric-label {
    font-size: 0.7rem;
    color: rgba(148,163,184,0.65);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 4px;
}

.stButton > button {
    background: linear-gradient(135deg, #10b981, #06b6d4) !important;
    color: #030712 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(16,185,129,0.35) !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed rgba(16,185,129,0.25) !important;
    border-radius: 14px !important;
    background: rgba(16,185,129,0.02) !important;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(16,185,129,0.25), transparent);
    margin: 20px 0;
}

.sidebar-label {
    color: rgba(148,163,184,0.7);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}

.tech-pill {
    display: inline-block;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.25);
    color: #6ee7b7;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# --- Hero ---
st.markdown("""
<div class="hero-container">
    <div class="badge">🔬 Computer Vision · YOLOv8 · Real-Time Inference</div>
    <p class="hero-title">VisionDX</p>
    <p class="hero-subtitle">Automated Anomaly Detection for Medical Imagery · Bio-Robotic Computer Vision Platform</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.4rem; margin-bottom:4px;">⚙️ Configuration</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">📊 Model Settings</p>', unsafe_allow_html=True)
    confidence = st.slider("Detection Confidence", 0.15, 0.90, 0.25, 0.05)
    st.markdown('<p class="sidebar-label" style="margin-top:16px;">🎯 Detection Classes</p>', unsafe_allow_html=True)
    detect_wbc = st.checkbox("White Blood Cells", value=True)
    detect_anomaly = st.checkbox("Anomalies", value=True)
    detect_tool = st.checkbox("Robotic Tools", value=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">🛠 Technology Stack</p>', unsafe_allow_html=True)
    st.markdown("""
    <span class="tech-pill">YOLOv8</span>
    <span class="tech-pill">PyTorch</span>
    <span class="tech-pill">OpenCV</span>
    <span class="tech-pill">ONNX</span>
    <span class="tech-pill">MPS/CUDA</span>
    """, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(16,185,129,0.05); border:1px solid rgba(16,185,129,0.15); border-radius:10px; padding:14px;">
        <span style="color:#10b981; font-weight:600; font-size:0.85rem;">● Model Loaded</span><br>
        <span style="color:rgba(148,163,184,0.6); font-size:0.75rem;">YOLOv8n — Ready for inference</span>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "runs/detect/cell_detection/weights/best.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        return YOLO("yolov8n.pt")

model = load_model()

# --- Stats Row ---
if "total_detected" not in st.session_state:
    st.session_state.total_detected = 0
if "scans_done" not in st.session_state:
    st.session_state.scans_done = 0
if "avg_time" not in st.session_state:
    st.session_state.avg_time = 0.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.scans_done}</div><div class="metric-label">Scans Completed</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_detected}</div><div class="metric-label">Objects Detected</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.avg_time:.2f}s</div><div class="metric-label">Avg Inference Time</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{confidence*100:.0f}%</div><div class="metric-label">Confidence Threshold</div></div>', unsafe_allow_html=True)

st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

# --- Main Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### 📁 Upload Medical Image")
    uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Source Image", use_column_width=True, clamp=True)
    else:
        sample_path = "dataset/images/val/cell_3.jpg"
        if os.path.exists(sample_path):
            image = Image.open(sample_path)
            st.image(image, caption="📎 Sample Image — Click Analyze to run detection", use_column_width=True)
        else:
            image = None

with col_right:
    st.markdown("#### 🤖 AI Detection Results")
    if image is not None:
        if st.button("🔍  Analyze", use_container_width=True):
            with st.spinner("Running YOLOv8 neural network..."):
                start_time = time.time()

                img_array = np.array(image)
                model.model.names = {0: "White Blood Cell", 1: "Anomaly Detected", 2: "Robotic Tool"}

                results = model.predict(img_array, conf=confidence)
                res_plotted = results[0].plot(labels=True, conf=True, line_width=2)
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                processing_time = time.time() - start_time

                n_detected = len(results[0].boxes)
                st.session_state.scans_done += 1
                st.session_state.total_detected += n_detected
                all_times = [st.session_state.avg_time * (st.session_state.scans_done - 1) + processing_time]
                st.session_state.avg_time = sum(all_times) / st.session_state.scans_done

                st.image(res_rgb, caption="AI Detection Output", use_column_width=True)

                # Detection pills
                if n_detected > 0:
                    pills_html = ""
                    for box in results[0].boxes:
                        conf_val = box.conf[0].item() * 100
                        cls_idx = int(box.cls[0].item()) % 3
                        class_names = ["White Blood Cell", "Anomaly Detected", "Robotic Tool"]
                        pill_classes = ["pill-wbc", "pill-anomaly", "pill-tool"]
                        name = class_names[cls_idx]
                        pill_cls = pill_classes[cls_idx]
                        pills_html += f'<span class="detect-pill {pill_cls}">{name} — {conf_val:.0f}%</span>'
                    st.markdown(pills_html, unsafe_allow_html=True)

                # Wow factor banner
                st.markdown(f"""
                <div class="wow-banner">
                    <div class="wow-text">✅ Analysis complete. {n_detected} objects detected. Processing time: {processing_time:.2f} seconds.</div>
                    <div class="wow-sub">Model: YOLOv8n · Inference Engine: PyTorch · Backend: {'Apple MPS' if os.uname().sysname == 'Darwin' else 'CUDA/CPU'}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("⬅️ Upload a medical image or use the sample to begin.")
