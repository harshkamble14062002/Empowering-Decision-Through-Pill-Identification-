import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to your dataset
dataset_dir = 'yolo_dataset'  # Update this path
images_dir = os.path.join(dataset_dir, 'yolo images')
labels_dir = os.path.join(dataset_dir, 'yolo labels')

# Output directories
train_images_dir = os.path.join(dataset_dir, 'yolo images2/train')
val_images_dir = os.path.join(dataset_dir, 'yolo images2/val')
train_labels_dir = os.path.join(dataset_dir, 'yolo labels2/train')
val_labels_dir = os.path.join(dataset_dir, 'yolo labels2/val')

# Create output directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# List all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Split dataset into training and validation
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Copy files to the respective directories
for file in train_files:
    shutil.copy(os.path.join(images_dir, file), os.path.join(train_images_dir, file))
    label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
    if os.path.exists(os.path.join(labels_dir, label_file)):
        shutil.copy(os.path.join(labels_dir, label_file), os.path.join(train_labels_dir, label_file))

for file in val_files:
    shutil.copy(os.path.join(images_dir, file), os.path.join(val_images_dir, file))
    label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
    if os.path.exists(os.path.join(labels_dir, label_file)):
        shutil.copy(os.path.join(labels_dir, label_file), os.path.join(val_labels_dir, label_file))
