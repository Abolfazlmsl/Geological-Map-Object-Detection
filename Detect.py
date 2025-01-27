#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:52:26 2024

@author: abolfazl
"""

import cv2
import os
from ultralytics import YOLO
import numpy as np

tileSize = 150
overlap = 50
threshold = 0.1
iou_threshold = 0.1

# Define colors for different classes
CLASS_COLORS = {
    0: (255, 0, 0),  # Landslide - Red
    1: (0, 255, 0),  # Strike - Green
    2: (0, 0, 255),  # Spring - Blue
    3: (255, 255, 0),  # Minepit - Yellow
    4: (255, 0, 255),  # Hillside - Pink
    5: (0, 255, 255),  # Feuchte - Light blue
    6: (0, 0, 0),  # Torf - Black
    7: (127, 127, 127),  # Bergsturz - Gray
    8: (50, 20, 60), # Landslide 2 - 
    9: (60, 50, 20), # Spring 2 - 
    10: (200, 150, 80), # Spring 3 - 
    11: (100, 200, 150), # Minepit 2 - 
    12: (12, 52, 83), # Spring B2 - 
}

# Define class names
CLASS_NAMES = {
    0: "Landslide 1",
    1: "Strike",
    2: "Spring 1",
    3: "Minepit 1",
    4: "Hillside",
    5: "Feuchte",
    6: "Torf",
    7: "Bergsturz",
    8: "Landslide 2",
    9: "Spring 2",
    10: "Spring 3",
    11: "Minepit 2",
    12: "Spring B2",
}

# Add threshold for each class
CLASS_THRESHOLDS = {
    0: 0.5,  # Landslide 1
    1: 0.5,  # Strike
    2: 0.5,  # Spring 1
    3: 0.5,  # Minepit 1
    4: 0.5,  # Hillside
    5: 0.5,  # Feuchte
    6: 0.05,  # Torf
    7: 0.5,  # Bergsturz
    8: 0.5,  # Landslide 2
    9: 0.05,  # Spring 2
    10: 0.05,  # Spring 3
    11: 0.5,  # Minepit 2
    12: 0.5,  # Spring B2
}

# Classes to exclude completely (will not be shown on the image)
EXCLUDED_CLASSES = {}  

def convert_to_grayscale(image):
    """
    Convert an image to grayscale and return a 3-channel grayscale image.

    Parameters:
    image: np.ndarray, input color image

    Returns:
    gray_3channel: np.ndarray, 3-channel grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray_3channel


def non_max_suppression(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.

    Parameters:
    detections: list of (x1, y1, x2, y2, cls, conf)
    iou_threshold: float, IOU threshold to filter overlapping boxes

    Returns:
    list of filtered detections
    """
    if len(detections) == 0:
        return []

    boxes = np.array([det[:4] for det in detections])
    scores = np.array([det[5] for det in detections])
    classes = np.array([det[4] for det in detections])

    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        remaining = indices[1:]
        ious = np.array([compute_iou(boxes[current], boxes[idx]) for idx in remaining])
        filtered_indices = np.where(ious < iou_threshold)[0]
        indices = remaining[filtered_indices]

    return [detections[k] for k in keep]


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IOU) of two bounding boxes.

    Parameters:
    box1, box2: (x1, y1, x2, y2)

    Returns:
    iou: float, intersection over union value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def detect_and_reconstruct(image_path, model, output_dir, tile_size=512, overlap=200, threshold=0.3):
    """
    Detect symbols in a large image, crop it into smaller tiles, 
    run YOLO detection on each tile, and then stitch the tiles back together.
    """
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    step = tile_size - overlap
    result_image = image.copy()
    detections = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            crop = image[y:y + tile_size, x:x + tile_size]
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                continue

            # Convert crop to grayscale using the new function
            crop_gray_3channel = convert_to_grayscale(crop)

            # Run YOLO detection on the grayscale image
            results = model(crop_gray_3channel)
            crop_detections = results[0].boxes

            for box in crop_detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in EXCLUDED_CLASSES:
                    continue

                if conf < CLASS_THRESHOLDS.get(cls, threshold):
                    continue

                adjusted_x1 = x1 + x
                adjusted_y1 = y1 + y
                adjusted_x2 = x2 + x
                adjusted_y2 = y2 + y

                detections.append((adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2, cls, conf))

    detections = non_max_suppression(detections, iou_threshold)

    for (x1, y1, x2, y2, cls, conf) in detections:
        color = CLASS_COLORS.get(cls, (0, 255, 255))
        label = CLASS_NAMES.get(cls, f"Class{cls}")

        # Draw rectangle around the detection on the original (color) image
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        # Put the class label and confidence score near the bounding box
        cv2.putText(
            result_image, f"{label} {conf:.2f}", (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, image_name.replace(".jpg", "_detected.jpg").replace(".png", "_detected.png"))
    cv2.imwrite(output_path, result_image)


# Initialize YOLO model
model = YOLO("best.pt")

# Define input and output directories
input_dir = "Input"
output_dir = "Output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all images in the input directory
for image_file in os.listdir(input_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, image_file)
        print(f"Processing {image_file}...")
        detect_and_reconstruct(image_path, model, output_dir, tile_size=tileSize, overlap=overlap, threshold=threshold)
        print(f"Saved results for {image_file} to {output_dir}")
