# steps/step03_morphology.py

import cv2
import numpy as np


# ---------------------------------------------------
# STEP 3 — MORPHOLOGY + AREA FILTERING
#
# Input:
# - Binary mask from STEP 2
#
# Output:
# - Closed mask
# - Noise-filtered mask
# ---------------------------------------------------

def morphology_filter(mask):

    # ------------------------------------------------
    # MORPHOLOGICAL CLOSING
    #
    # Purpose:
    # - Fill small holes
    # - Connect broken regions
    # ------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # ------------------------------------------------
    # CONNECTED COMPONENT ANALYSIS
    #
    # Purpose:
    # - Remove tiny noisy blobs
    # ------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed,
        connectivity=8
    )

    filtered = np.zeros_like(closed)

    # ------------------------------------------------
    # Keep only sufficiently large regions
    # ------------------------------------------------

    MIN_AREA = 120

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > MIN_AREA:

            filtered[labels == i] = 255

    return closed, filtered