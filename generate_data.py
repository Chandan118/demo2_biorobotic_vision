import os
import random
import cv2
import numpy as np
import yaml

def generate_synthetic_dataset(base_dir="dataset", num_images=20):
    images_dir = os.path.join(base_dir, "images/train")
    labels_dir = os.path.join(base_dir, "labels/train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Validation dirs
    val_images_dir = os.path.join(base_dir, "images/val")
    val_labels_dir = os.path.join(base_dir, "labels/val")
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    def create_image(img_dir, lbl_dir, idx):
        # Create a blank white image (simulating microscope slide)
        img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        
        num_cells = random.randint(1, 5)
        boxes = []
        for _ in range(num_cells):
            # Random cell properties
            r = random.randint(15, 40)
            x = random.randint(r, 512-r)
            y = random.randint(r, 512-r)
            color = (random.randint(100, 200), random.randint(0, 100), random.randint(100, 200)) # Purple/Pinkish
            
            # Draw cell
            cv2.circle(img, (x, y), r, color, -1)
            cv2.circle(img, (x, y), r, (0,0,0), 2)
            
            # YOLO format: class x_center y_center width height (normalized)
            x_center = x / 512.0
            y_center = y / 512.0
            width = (r * 2) / 512.0
            height = (r * 2) / 512.0
            boxes.append(f"0 {x_center} {y_center} {width} {height}")
            
        cv2.imwrite(os.path.join(img_dir, f"cell_{idx}.jpg"), img)
        with open(os.path.join(lbl_dir, f"cell_{idx}.txt"), "w") as f:
            f.write("\n".join(boxes))

    print(f"Generating {num_images} training images...")
    for i in range(num_images):
        create_image(images_dir, labels_dir, i)
        
    print(f"Generating {num_images // 4} validation images...")
    for i in range(num_images // 4):
        create_image(val_images_dir, val_labels_dir, i)
        
    # Create dataset.yaml
    dataset_yaml = {
        'path': os.path.abspath(base_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'Cell'}
    }
    
    with open("dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)
    print("Dataset generation complete. Wrote dataset.yaml")

if __name__ == "__main__":
    generate_synthetic_dataset()
