# utils/io.py
# Input/Output utilities for image processing

import os
import cv2
import numpy as np
from config import INPUT_DIR, OUTPUT_DIR

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

def get_image_list(input_dir=INPUT_DIR):
    """Get list of image files in input directory."""
    if not os.path.exists(input_dir):
        return []
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in extensions):
            images.append(os.path.join(input_dir, file))
    return images