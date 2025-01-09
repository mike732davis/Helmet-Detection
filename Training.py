import os
from ultralytics import YOLO


# 1. Define Dataset Paths
train_path = 'helmetDataset/train/images'  # Path to training images
val_path = 'helmetDataset/val/images'  # Path to validation images
labels_train_path = 'helmetDataset/train/labels'  # Path to training labels
labels_val_path = 'helmetDataset/val/labels'  # Path to validation labels

# 2. Create data.yaml (for YOLO)
data_yaml = """
train: G:/HelmetDetection/helmetDataset/train/images
val: G:/HelmetDetection/helmetDataset/val/images

nc: 2
names: ['With Helmet', 'Without Helmet']
"""

# Save the data.yaml file
with open('helmet_data.yaml', 'w') as file:
    file.write(data_yaml)

# 3. Train YOLOv8 Model on Your Dataset
# Load the YOLOv8 model (pretrained)
model = YOLO('yolov8n.pt')  # You can replace with yolov8s.pt or yolov8m.pt for higher accuracy

# Train the model
model.train(data='helmet_data.yaml', epochs=5, imgsz=640)
