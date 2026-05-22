import cv2


# ---------------------------------------------------
# STEP 1 — ROI EXTRACTION
# ---------------------------------------------------

def extract_roi(image):

    h, w = image.shape[:2]

    # Preserve upper-right roadside region

    x_start = int(w * 0.50)

    y_start = 0
    y_end = int(h * 0.85)

    roi = image[
        y_start:y_end,
        x_start:w
    ]

    return roi