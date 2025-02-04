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
import pandas as pd

tile_sizes = [150, 400]
overlaps = [100, 250]
iou_threshold = 0.01
models = [YOLO("best150.pt"), YOLO("best400.pt")]

# Define colors for different classes
CLASS_COLORS = {
    0: (255, 0, 0),  # Landslide
    1: (0, 255, 0),  # Strike
    2: (0, 0, 255),  # Spring
    3: (255, 255, 0),  # Minepit
    4: (255, 0, 255),  # Hillside
    5: (0, 255, 255),  # Feuchte
    6: (0, 0, 0),  # Torf
    7: (127, 127, 127),  # Bergsturz
    8: (50, 20, 60), # Landslide 2  
    9: (60, 50, 20), # Spring 2  
    10: (200, 150, 80), # Spring 3  
    11: (100, 200, 150), # Minepit 2 
    12: (12, 52, 83), # Spring B2 
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
    0: 0.6,  # Landslide 1
    1: 0.7,  # Strike
    2: 0.6,  # Spring 1
    3: 0.6,  # Minepit 1
    4: 0.7,  # Hillside
    5: 0.05,  # Feuchte
    6: 0.05,  # Torf
    7: 0.05,  # Bergsturz
    8: 0.05,  # Landslide 2
    9: 0.05,  # Spring 2
    10: 0.05,  # Spring 3
    11: 0.4,  # Minepit 2
    12: 0.05,  # Spring B2
}

# Classes to exclude completely (will not be shown on the image)
EXCLUDED_CLASSES = {12}  

# def calculate_angle(x1, y1, x2, y2):
#     dx = x2 - x1
#     dy = y2 - y1
#     angle = np.arctan2(dy, dx) * (180.0 / np.pi)
#     return angle if angle >= 0 else angle + 180

# def find_main_strike_angle(edges):
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=5, maxLineGap=5)
    
#     if lines is None:
#         return None 
    
#     longest_line = None
#     max_length = 0

#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) 
#         if length > max_length:
#             max_length = length
#             longest_line = (x1, y1, x2, y2)

#     if longest_line:
#         x1, y1, x2, y2 = longest_line
#         angle = np.arctan2(y2 - y1, x2 - x1) * (180.0 / np.pi)  
#         vertical_angle = (90 - angle) % 180  
#         return vertical_angle

#     return None


def convert_to_grayscale(image):
    """
    Convert an image to grayscale and return a 3-channel grayscale image.

    Parameters:
    image (np.ndarray): Input color image.

    Returns:
    np.ndarray: 3-channel grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    box1 (tuple): Coordinates of the first bounding box (x1, y1, x2, y2).
    box2 (tuple): Coordinates of the second bounding box (x1, y1, x2, y2).

    Returns:
    float: IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area)

def detect_symbols(image, model, tile_size, overlap):
    """
    Detect objects in an image using a given model, tile size, and overlap.

    Parameters:
    image (np.ndarray): Input image.
    model (YOLO): YOLO model used for detection.
    tile_size (int): Size of the detection tile.
    overlap (int): Overlap between tiles.

    Returns:
    list: Detected bounding boxes with format (x1, y1, x2, y2, class, confidence).
    """
    h, w, _ = image.shape
    step = tile_size - overlap
    detections = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            crop = image[y:y + tile_size, x:x + tile_size]
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                continue
            crop_gray = convert_to_grayscale(crop)
            results = model(crop_gray)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in EXCLUDED_CLASSES or conf < CLASS_THRESHOLDS.get(cls, 0.05):
                    continue
                detections.append((x1 + x, y1 + y, x2 + x, y2 + y, cls, conf))
    return detections

def merge_detections(detections, iou_threshold=0.5):
    """
    Merge overlapping detections while considering confidence and class types.

    Parameters:
    detections (list): List of detected bounding boxes (x1, y1, x2, y2, class, confidence).
    iou_threshold (float): IoU threshold for merging overlapping detections.

    Returns:
    list: Filtered list of detections.
    """
    if not detections:
        return []
    
    detections.sort(key=lambda x: x[5], reverse=True)
    merged = []

    for i, det1 in enumerate(detections):
        
        x1_1, y1_1, x2_1, y2_1, cls1, conf1 = det1
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        keep = True

        for j, det2 in enumerate(detections):
            if i == j:
                continue
            
            x1_2, y1_2, x2_2, y2_2, cls2, conf2 = det2
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            iou = compute_iou(det1[:4], det2[:4])

            if iou >= iou_threshold:
                if cls1 == cls2:  
                    intersection_x1 = max(x1_1, x1_2)
                    intersection_y1 = max(y1_1, y1_2)
                    intersection_x2 = min(x2_1, x2_2)
                    intersection_y2 = min(y2_1, y2_2)
                    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
                    non_overlap_area = area1 - intersection_area
                    
                    if conf1 < conf2:
                        if non_overlap_area / area1 >= 0.4:
                            keep = True
                        else:
                            keep = False
                            break

        if keep:
            merged.append(det1)

    return merged

def process_image(image_path, output_dir):
    """
    Process an image by applying object detection at multiple tile sizes and merging the results.

    Parameters:
    image_path (str): Path to the input image.
    output_dir (str): Directory where the processed image will be saved.

    Returns:
    None
    """
    image = cv2.imread(image_path)
    all_detections = []
    for tile_size, overlap, model in zip(tile_sizes, overlaps, models):
        all_detections.extend(detect_symbols(image, model, tile_size, overlap))
    merged_detections = merge_detections(all_detections, iou_threshold)
    result_image = image.copy()
    # image_name = os.path.basename(image_path)
    # excel_path = os.path.join(output_dir, image_name.replace(".jpg", ".xlsx").replace(".png", ".xlsx"))

    # data = []
    for x1, y1, x2, y2, cls, conf in merged_detections:
        color = CLASS_COLORS.get(cls, (0, 255, 255))
        label = CLASS_NAMES.get(cls, f"Class{cls}")
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result_image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # width, height = x2 - x1, y2 - y1
        # angle = calculate_angle(x1, y1, x2, y2) if cls == 1 else None
        # if cls == 1:  
        #     roi = image[y1:y2, x1:x2]  
        #     edges = cv2.Canny(roi, 50, 150)  
        #     angle = find_main_strike_angle(edges)
        # else:
        #     angle = None
        # data.append([CLASS_NAMES.get(cls, f"Class{cls}"), x1, y1, x2, y2, width, height, conf, angle])

    output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_detected.jpg").replace(".png", "_detected.png"))
    cv2.imwrite(output_path, result_image)
    
    # df = pd.DataFrame(data, columns=["Class", "X1", "Y1", "X2", "Y2", "Width", "Height", "Confidence", "Angle"])
    # df.to_excel(excel_path, index=False)

# Define input and output directories
input_dir = "Input"
output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)

# Process all images in the input directory
for image_file in os.listdir(input_dir):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        print(f"Processing {image_file}...")
        process_image(os.path.join(input_dir, image_file), output_dir)
        print(f"Results saved for {image_file}")
