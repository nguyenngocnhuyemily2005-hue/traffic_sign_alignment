# utils/io.py
# Input/Output utilities for the Traffic Sign Alignment Pipeline


import os
import cv2

from config import OUTPUT_DIR


# ---------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ---------------------------------------------------

def create_output_dir():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ---------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------

def load_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(image_path)

    if image is None:

        raise ValueError(
            f"Could not load image: {image_path}"
        )

    return image


# ---------------------------------------------------
# SAVE IMAGE
# ---------------------------------------------------

def save_image(filename, image):

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(
        output_path,
        image
    )