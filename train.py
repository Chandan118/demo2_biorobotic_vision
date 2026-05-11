from ultralytics import YOLO
import platform
import torch

def train_model():
    print(f"System: {platform.system()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check if MPS (Metal Performance Shaders) is available for Mac M2
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train the model
    print("Starting training on synthetic dataset...")
    results = model.train(
        data='dataset.yaml',
        epochs=5,  # Short training for demo purposes
        imgsz=512,
        device=device,
        project='runs/detect',
        name='cell_detection',
        exist_ok=True
    )
    print("Training complete. Model saved in runs/detect/cell_detection/weights/best.pt")

if __name__ == "__main__":
    train_model()
