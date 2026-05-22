import cv2
import numpy as np


# ---------------------------------------------------
# STEP 4 — CONTOUR EXTRACTION
# ---------------------------------------------------

def extract_contours(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


# ---------------------------------------------------
# DRAW CONTOURS + BOUNDING BOXES
# ---------------------------------------------------

def draw_contours(image, contours):

    output = image.copy()

    for cnt in contours:

        # Draw contour

        cv2.drawContours(
            output,
            [cnt],
            -1,
            (0, 255, 0),
            2
        )

        # Bounding box

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    return output