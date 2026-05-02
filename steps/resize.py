# steps/resize.py
# Image resizing step

import cv2
from config import DEFAULT_IMAGE_SIZE

def resize_image(image, size=DEFAULT_IMAGE_SIZE):
    """Resize an image to the specified size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def resize_with_aspect_ratio(image, max_size):
    """Resize image maintaining aspect ratio."""
    height, width = image.shape[:2]
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)