# utils/io.py
# Input/Output utilities for image processing

import os
import cv2
import numpy as np
from config import INPUT_DIR, LABELS_DIR, OUTPUT_DIR

def load_image(image_path):
    """Load an image from file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image

def save_image(image, filename, output_dir=OUTPUT_DIR):
    """Save an image to file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    return output_path

def get_image_list(input_dir=INPUT_DIR, labels_dir=LABELS_DIR):
    """Get list of (image_path, label_path) pairs. label_path is None if no label exists."""
    if not os.path.exists(input_dir):
        return []
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    pairs = []
    for file in sorted(os.listdir(input_dir)):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_path = os.path.join(input_dir, file)
            stem = os.path.splitext(file)[0]
            label_path = os.path.join(labels_dir, stem + '.txt')
            pairs.append((image_path, label_path if os.path.exists(label_path) else None))
    return pairs

def load_label(label_path):
    """Load YOLO-format label file. Returns list of (class_id, cx, cy, w, h) tuples."""
    if label_path is None or not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append((int(parts[0]), float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4])))
    return labels