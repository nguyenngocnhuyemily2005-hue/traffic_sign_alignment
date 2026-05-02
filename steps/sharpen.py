# steps/sharpen.py
# Image sharpening step

import cv2
import numpy as np
from config import SHARPEN_FACTOR

def sharpen_image(image, factor=SHARPEN_FACTOR):
    """Sharpen an image using unsharp masking."""
    # Create a Gaussian blur
    gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
    # Calculate the unsharp mask
    unsharp_mask = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    # Apply sharpening
    sharpened = cv2.addWeighted(image, 1 + factor, unsharp_mask, -factor, 0)
    return sharpened

def sharpen_with_kernel(image):
    """Sharpen using a custom kernel."""
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)