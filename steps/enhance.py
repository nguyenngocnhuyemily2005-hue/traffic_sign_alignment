# steps/enhance.py
# Image enhancement step

import cv2
import numpy as np
from config import ENHANCE_CONTRAST

def enhance_contrast(image, factor=ENHANCE_CONTRAST):
    """Enhance image contrast."""
    # Convert to LAB color space for better contrast adjustment
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8,8))
    l = clahe.apply(l)
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_brightness(image, value=30):
    """Adjust image brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)