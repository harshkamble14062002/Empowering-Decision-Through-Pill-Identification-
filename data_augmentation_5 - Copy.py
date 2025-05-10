import cv2
import albumentations as A
import os
import glob

# Define augmentation pipeline with added transformations
transform = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # Apply Gaussian blur
    A.RandomBrightnessContrast(p=0.7),  # Adjust brightness/contrast
    A.RandomScale(scale_limit=0.2, p=0.5),  # Scale the image randomly
    A.Resize(640, 640),  # Resize image to 640x640, required for YOLO model compatibility
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_yolo_annotation(txt_file):
    bboxes = []
    class_labels = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))
    return bboxes, class_labels

def save_yolo_annotation(txt_file, bboxes, class_labels):
    with open(txt_file, 'w') as f:
        for bbox, label in zip(bboxes, class_labels):
            f.write(f"{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

# Input and output directories
input_img_dir = 'yolo_dataset/yolo images'  # Folder containing original images
input_label_dir = 'yolo_dataset/yolo labels'  # Folder containing original .txt files
output_img_dir = 'augmented_images'  # Folder to save augmented images
output_label_dir = 'aug_labels'  # Folder to save new .txt files

# Create directories for augmented data
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Iterate through all images and their corresponding labels
for img_file in glob.glob(f'{input_img_dir}/*.jpg'):  # Adjust based on your image format
    img_name = os.path.basename(img_file)
    txt_file = os.path.join(input_label_dir, img_name.replace('.jpg', '.txt'))

    # Read the image and its bounding boxes
    image = cv2.imread(img_file)
    bboxes, class_labels = read_yolo_annotation(txt_file)

    # Apply augmentations
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_class_labels = augmented['class_labels']

    # Save the augmented image and corresponding labels
    cv2.imwrite(os.path.join(output_img_dir, img_name), augmented_image)
    save_yolo_annotation(os.path.join(output_label_dir, img_name.replace('.jpg', '.txt')), augmented_bboxes, augmented_class_labels)