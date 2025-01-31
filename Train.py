#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:52:26 2024

@author: abolfazl
"""

import os
import cv2
import pandas as pd
import numpy as np
import random
import torch
from ultralytics import YOLO

# Configuration
need_cropping = False
need_augmentation = False
tile_size = 64
overlap = 32
epochs = 50
batch_size = 32
object_boundary_threshold = 0.1  # Minimum fraction of the bounding box that must remain in the crop
class_balance_threshold = 500  # Minimum number of samples per class for balance
augmentation_repeats = 5  # Number of times to augment underrepresented classes

def update_txt_file(txt_file, new_paths):
    """
    Update the .txt file with new paths of cropped or augmented images.
    """
    with open(txt_file, "w") as f:
        for path in new_paths:
            f.write(f"{path}\n")

def convert_to_grayscale(image):
    """
    Convert an image to grayscale and ensure it has 3 channels.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Ensure 3-channel format for consistency
    return gray_image

def crop_images_and_labels(image_dir, label_dir, output_image_dir, output_label_dir, txt_file, cropped_txt_file, tile_size=512, overlap=0):
    """
    Crop images and adjust labels for YOLO format. Save results and update the .txt file with new image paths.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    new_paths = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_file}")
            continue

        # Convert image to grayscale
        image = convert_to_grayscale(image)
        
        h, w, _ = image.shape
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Label file not found: {label_file}")
            continue

        labels = pd.read_csv(label_path, sep=" ", header=None)
        labels.columns = ["class", "x_center", "y_center", "width", "height"]
        labels["x_center"] *= w
        labels["y_center"] *= h
        labels["width"] *= w
        labels["height"] *= h

        step = tile_size - overlap
        tile_id = 0
        for y in range(0, h, step):
            for x in range(0, w, step):
                crop = image[y:y + tile_size, x:x + tile_size]
                if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                    continue

                # Find labels within the crop region
                tile_labels = labels[
                    (labels["x_center"] >= x) & (labels["x_center"] < x + tile_size) &
                    (labels["y_center"] >= y) & (labels["y_center"] < y + tile_size)
                ].copy()

                # Adjust coordinates of labels for the crop
                tile_labels["x_center"] -= x
                tile_labels["y_center"] -= y

                # Filter out bounding boxes that are partially outside the crop
                valid_labels = []
                for _, row in tile_labels.iterrows():
                    x1 = row["x_center"] - row["width"] / 2
                    y1 = row["y_center"] - row["height"] / 2
                    x2 = row["x_center"] + row["width"] / 2
                    y2 = row["y_center"] + row["height"] / 2

                    # Calculate intersection area between the box and the crop
                    inter_x1 = max(x1, 0)
                    inter_y1 = max(y1, 0)
                    inter_x2 = min(x2, tile_size)
                    inter_y2 = min(y2, tile_size)
                    intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    original_area = (x2 - x1) * (y2 - y1)

                    # Keep the box if the intersection area is above a threshold
                    if intersection_area / original_area >= object_boundary_threshold:
                        row["x_center"] = (inter_x1 + inter_x2) / 2
                        row["y_center"] = (inter_y1 + inter_y2) / 2
                        row["width"] = inter_x2 - inter_x1
                        row["height"] = inter_y2 - inter_y1
                        valid_labels.append(row)

                # Skip saving this crop if no valid labels remain
                if not valid_labels:
                    continue

                valid_labels_df = pd.DataFrame(valid_labels)

                # Normalize the adjusted labels for YOLO format
                valid_labels_df["x_center"] /= tile_size
                valid_labels_df["y_center"] /= tile_size
                valid_labels_df["width"] /= tile_size
                valid_labels_df["height"] /= tile_size

                # Save cropped image
                tile_image_filename = f"{os.path.splitext(image_file)[0]}_tile_{tile_id}.jpg"
                tile_image_path = os.path.join(output_image_dir, tile_image_filename)
                cv2.imwrite(tile_image_path, crop)

                # Save adjusted labels
                tile_label_filename = f"{os.path.splitext(image_file)[0]}_tile_{tile_id}.txt"
                tile_label_path = os.path.join(output_label_dir, tile_label_filename)
                valid_labels_df.to_csv(tile_label_path, sep=" ", header=False, index=False)

                # Store new image path for updating the txt file
                new_paths.append(tile_image_path)

                tile_id += 1

        print(f"Processed image: {image_file}")

    update_txt_file(cropped_txt_file, new_paths)

def apply_single_class_augmentation(image, labels, target_class):
    """
    Apply augmentations to an image and labels, targeting a specific class.
    """
    aug_image = image.copy()
    aug_labels = labels.copy()

    # Select only target class labels
    target_labels = aug_labels[aug_labels["class"] == target_class]

    # Apply random scaling
    if random.random() > 0.5:
        scale_factor = random.uniform(0.8, 1.2)  # Random scale between 80% to 120%
        height, width = aug_image.shape[:2]
        aug_image = cv2.resize(aug_image, (int(width * scale_factor), int(height * scale_factor)))

        # Adjust label coordinates based on scaling
        target_labels["x_center"] *= scale_factor
        target_labels["y_center"] *= scale_factor
        target_labels["width"] *= scale_factor
        target_labels["height"] *= scale_factor

    # Apply random noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.5, aug_image.shape).astype(np.uint8)  # Gaussian noise
        aug_image = cv2.add(aug_image, noise)

    # Change brightness and contrast
    if random.random() > 0.5:
        aug_image = cv2.convertScaleAbs(aug_image, alpha=1.2, beta=50)

    aug_labels.update(target_labels)
    return aug_image, aug_labels

def update_balanced_txt_file(txt_file, new_paths):
    """
    Append new paths of augmented images to the .txt file.
    """
    with open(txt_file, "a") as f:  # Append mode
        for path in new_paths:
            f.write(f"{path}\n")

def balance_classes(image_dir, label_dir, txt_file):
    """
    Balance classes by oversampling underrepresented classes with augmentations,
    and update the txt file with new image paths.
    """
    # Calculate class distribution
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    class_counts = {}
    for label_file in label_files:
        labels = pd.read_csv(os.path.join(label_dir, label_file), sep=" ", header=None)
        labels.columns = ["class", "x_center", "y_center", "width", "height"]
        for class_id in labels["class"]:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

    print(f"Initial class distribution: {class_counts}")

    new_image_paths = []

    for class_id, count in class_counts.items():
        if count >= class_balance_threshold:
            continue

        print(f"Balancing class {class_id} (current count: {count})")

        images_with_class = []

        for label_file in label_files:
            file_path = os.path.join(label_dir, label_file)

            labels = pd.read_csv(file_path, sep=" ", header=None)
            if labels.shape[1] != 5:
                print(f"Invalid format in file {label_file}")
                continue
            labels.columns = ["class", "x_center", "y_center", "width", "height"]
            
            if class_id in labels["class"].values:
                images_with_class.append(label_file)

        for _ in range(augmentation_repeats):
            for label_file in images_with_class:
                image_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))
                image = cv2.imread(image_path)

                if image is None:
                    continue

                labels = pd.read_csv(os.path.join(label_dir, label_file), sep=" ", header=None)
                labels.columns = ["class", "x_center", "y_center", "width", "height"]

                aug_image, aug_labels = apply_single_class_augmentation(image, labels, class_id)

                # Save augmented image
                aug_image_filename = f"{os.path.splitext(label_file)[0]}_aug_{random.randint(0, 10000)}.jpg"
                aug_image_path = os.path.join(image_dir, aug_image_filename)
                cv2.imwrite(aug_image_path, aug_image)

                # Save augmented labels
                aug_label_filename = f"{os.path.splitext(label_file)[0]}_aug_{random.randint(0, 10000)}.txt"
                aug_label_path = os.path.join(label_dir, aug_label_filename)
                aug_labels.to_csv(aug_label_path, sep=" ", header=False, index=False)

                # Add new image path to list for txt file
                new_image_paths.append(aug_image_path)

    # Update the txt file with new paths
    update_balanced_txt_file(txt_file, new_image_paths)

    print(f"Balanced class distribution: {class_counts}")


if __name__ == "__main__":
    
    image_dir = "datasets/GeoMap/images/train"
    label_dir = "datasets/GeoMap/labels/train"
    output_image_dir = "datasets/GeoMap/cropped/images/train"
    output_label_dir = "datasets/GeoMap/cropped/labels/train"
    txt_file = "datasets/GeoMap/train.txt"
    cropped_txt_file = "datasets/GeoMap/train_cropped.txt"

    val_image_dir = "datasets/GeoMap/images/val"
    val_label_dir = "datasets/GeoMap/labels/val"
    val_output_image_dir = "datasets/GeoMap/cropped/images/val"
    val_output_label_dir = "datasets/GeoMap/cropped/labels/val"
    val_txt_file = "datasets/GeoMap/val.txt"
    val_cropped_txt_file = "datasets/GeoMap/val_cropped.txt"

    if need_cropping:
        crop_images_and_labels(
            image_dir=image_dir,
            label_dir=label_dir,
            output_image_dir=output_image_dir,
            output_label_dir=output_label_dir,
            txt_file=txt_file,
            cropped_txt_file=cropped_txt_file,
            tile_size=tile_size,
            overlap=overlap,
        )

        crop_images_and_labels(
            image_dir=val_image_dir,
            label_dir=val_label_dir,
            output_image_dir=val_output_image_dir,
            output_label_dir=val_output_label_dir,
            txt_file=val_txt_file,
            cropped_txt_file=val_cropped_txt_file,
            tile_size=tile_size,
            overlap=overlap,
        )

    if need_augmentation:
        balance_classes(
            image_dir=output_image_dir,
            label_dir=output_label_dir,
            txt_file=cropped_txt_file,
        )

        balance_classes(
            image_dir=val_output_image_dir,
            label_dir=val_output_label_dir,
            txt_file=val_cropped_txt_file,
        )

    model = YOLO("yolo11x.pt")

    model.train(
        data="datasets/GeoMap/data.yaml",
        epochs=epochs,
        imgsz=tile_size,  # Image size (same as crop size)
        batch=batch_size,
        multi_scale=True,
        device=[0, 1] if torch.cuda.is_available() else "CPU",
    )