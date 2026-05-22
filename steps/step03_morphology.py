import cv2
import numpy as np


# ---------------------------------------------------
# STEP 3 — MORPHOLOGY
# ---------------------------------------------------

def apply_closing(mask):

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closed


# ---------------------------------------------------
# AREA FILTERING
# ---------------------------------------------------

def area_filter(mask, min_area=100):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered = np.zeros_like(mask)

    kept_contours = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area >= min_area:

            cv2.drawContours(
                filtered,
                [cnt],
                -1,
                255,
                thickness=cv2.FILLED
            )

            kept_contours.append(cnt)

    return filtered, kept_contours