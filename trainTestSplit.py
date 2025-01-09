import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Define paths for images and labels
images_dir = "helmetDataset/images"
labels_dir = "helmetDataset/labels"
train_images_dir = "helmetDataset/train/images"
val_images_dir = "helmetDataset/val/images"
train_labels_dir = "helmetDataset/train/labels"
val_labels_dir = "helmetDataset/val/labels"

# Create directories for train and val
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get list of all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]

# Get corresponding label files (assuming label files have the same name as image files)
label_files = [f.replace(".png", ".txt") for f in image_files]

# Split dataset into 80% train and 20% validation
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# Function to copy files to train/val directories
def copy_files(file_list, src_dir, dest_dir):
    for file in file_list:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

# Copy image files to train and val sets
copy_files(train_images, images_dir, train_images_dir)
copy_files(val_images, images_dir, val_images_dir)

# Copy label files to train and val sets
copy_files([f.replace(".png", ".txt") for f in train_images], labels_dir, train_labels_dir)
copy_files([f.replace(".png", ".txt") for f in val_images], labels_dir, val_labels_dir)

# Output the number of files in each set
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
