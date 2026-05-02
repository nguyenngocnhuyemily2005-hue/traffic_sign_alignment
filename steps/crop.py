# steps/crop.py
# Image cropping step

import cv2
import numpy as np

def crop_image(image, x, y, width, height):
    """Crop an image to the specified rectangle."""
    return image[y:y+height, x:x+width]

def auto_crop(image):
    """Automatically crop the image to remove borders."""
    # Simple auto-crop: remove 10% from each side
    height, width = image.shape[:2]
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    return crop_image(image, margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)