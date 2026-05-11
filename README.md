# 🔬 Bio-Robotic Computer Vision

An advanced computer vision system tailored for Bio-Robotics, capable of automating the detection of cells, anomalies, or surgical tools from microscope slides and camera feeds.

**Developed by Chandan Sheikder, PhD Researcher.**

![UI Mockup](https://via.placeholder.com/800x400.png?text=Bio-Robotic+Computer+Vision+Dashboard)

## 📌 The Problem
In medical labs and robotic surgeries, identifying microscopic anomalies or tracking surgical tools manually is slow and prone to human error. Automation requires state-of-the-art vision models that can process images instantly.

## 💡 The Solution
This AI leverages the Ultralytics YOLOv8 architecture to perform real-time object detection. The model can be trained on custom datasets (e.g., Blood Cell Count Dataset) and instantly draw bounding boxes around detected objects with high confidence.

## 🛠 Tech Stack
- **Language:** Python
- **AI Framework:** Ultralytics YOLOv8
- **Compute:** PyTorch (Accelerated via Apple Silicon MPS backend)
- **Frontend UI:** Streamlit

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Training Data & Train
We've included a script to generate a synthetic biological dataset (simulated cells) to demonstrate the training pipeline without requiring API keys.
```bash
python generate_data.py
python train.py
```
*Note: Due to M2 Pro optimization, training takes only seconds.*

### 3. Run the Web App
```bash
streamlit run app.py
```
Upload one of the generated images from `dataset/images/val/` to see the automated detection in action!
