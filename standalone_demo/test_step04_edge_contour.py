import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


# ===================================================
# STEP 4 — SEMANTIC CONTOUR FILTERING
#
# Pipeline:
# STEP 1 → ROI
# STEP 2 → HSV filtering
# STEP 3 → Morphology + Area filtering
# STEP 4 → Semantic contour filtering
#
# Goal:
# Detect traffic-sign-like objects only
# ===================================================


# ===================================================
# STEP 1 — SIMPLE ROI
# ===================================================

def extract_roi(image):

    h, w = image.shape[:2]

    x_start = int(w * 0.50)

    y_start = 0

    y_end = int(h * 0.85)

    roi = image[
        y_start:y_end,
        x_start:w
    ]

    return roi


# ===================================================
# STEP 2 — HSV FILTERING
# ===================================================

def hsv_filter(roi):

    blurred = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # ------------------------------------------------
    # BLUE MASK
    # ------------------------------------------------

    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    # ------------------------------------------------
    # RED MASK
    # ------------------------------------------------

    lower_red1 = np.array([0, 140, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 140, 80])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    red_mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    red_mask = cv2.add(
        red_mask1,
        red_mask2
    )

    return blue_mask, red_mask


# ===================================================
# STEP 3 — MORPHOLOGY
# ===================================================

def clean_mask(mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # -----------------------------------------------
    # AREA FILTERING
    # -----------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed,
        connectivity=8
    )

    filtered = np.zeros_like(closed)

    MIN_AREA = 120

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > MIN_AREA:

            filtered[labels == i] = 255

    return filtered


# ===================================================
# STEP 4 — SEMANTIC CONTOUR FILTERING
# ===================================================

def semantic_contours(mask, roi):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = roi.copy()

    detected = 0

    roi_h, roi_w = roi.shape[:2]

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # -------------------------------------------
        # BASIC AREA FILTER
        # -------------------------------------------

        if area < 250:
            continue

        perimeter = cv2.arcLength(
            cnt,
            True
        )

        if perimeter == 0:
            continue

        # -------------------------------------------
        # SHAPE APPROXIMATION
        # -------------------------------------------

        epsilon = 0.02 * perimeter

        approx = cv2.approxPolyDP(
            cnt,
            epsilon,
            True
        )

        corners = len(approx)

        # -------------------------------------------
        # BOUNDING BOX
        # -------------------------------------------

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / float(h)

        fill_ratio = area / float(w * h)

        # -------------------------------------------
        # CIRCULARITY
        # -------------------------------------------

        circularity = (
            4 * np.pi * area
        ) / (perimeter * perimeter)

        # -------------------------------------------
        # POSITION FILTERING
        # Traffic signs usually:
        # - upper/middle region
        # - right side
        # -------------------------------------------

        center_x = x + w // 2
        center_y = y + h // 2

        if center_y > roi_h * 0.90:
            continue

        if center_x < roi_w * 0.10:
            continue

        # -------------------------------------------
        # REMOVE THIN OBJECTS
        # like poles or cables
        # -------------------------------------------

        if aspect_ratio < 0.35:
            continue

        # -------------------------------------------
        # REMOVE EMPTY OBJECTS
        # -------------------------------------------

        if fill_ratio < 0.20:
            continue

        # ===================================================
        # SHAPE CLASSIFICATION
        # ===================================================

        shape_type = "unknown"

        color = (255, 0, 0)

        # -------------------------------------------
        # TRIANGLE
        # -------------------------------------------

        if corners == 3:

            shape_type = "triangle"

            color = (0, 255, 0)

        # -------------------------------------------
        # RECTANGLE
        # -------------------------------------------

        elif corners == 4:

            if 0.5 <= aspect_ratio <= 2.5:

                shape_type = "rectangle"

                color = (255, 0, 0)

        # -------------------------------------------
        # CIRCLE
        # -------------------------------------------

        elif circularity > 0.65:

            shape_type = "circle"

            color = (0, 0, 255)

        # -------------------------------------------
        # REJECT UNKNOWN SHAPES
        # -------------------------------------------

        else:
            continue

        # ===================================================
        # DRAW RESULTS
        # ===================================================

        detected += 1

        cv2.drawContours(
            output,
            [approx],
            -1,
            color,
            3
        )

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            output,
            shape_type,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return output, detected


# ===================================================
# MAIN
# ===================================================

os.makedirs(
    'output_images',
    exist_ok=True
)

input_folder = 'input_images'

image_files = os.listdir(input_folder)


for file_name in image_files:

    image_path = os.path.join(
        input_folder,
        file_name
    )

    img = cv2.imread(image_path)

    if img is None:

        print(f"Cannot load: {file_name}")

        continue

    # ===================================================
    # STEP 1
    # ===================================================

    roi = extract_roi(img)

    # ===================================================
    # STEP 2
    # ===================================================

    blue_mask, red_mask = hsv_filter(roi)

    # ===================================================
    # STEP 3
    # ===================================================

    blue_clean = clean_mask(blue_mask)

    red_clean = clean_mask(red_mask)

    final_mask = cv2.add(
        blue_clean,
        red_clean
    )

    # ===================================================
    # STEP 4
    # ===================================================

    contour_output, detected = semantic_contours(
        final_mask,
        roi
    )

    # ===================================================
    # SAVE
    # ===================================================

    cv2.imwrite(
        f'output_images/semantic_{file_name}',
        contour_output
    )

    # ===================================================
    # VISUALIZATION
    # ===================================================

    plt.figure(figsize=(18, 5))

    # ROI
    plt.subplot(1, 3, 1)

    plt.imshow(
        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    )

    plt.title("ROI")

    plt.axis('off')

    # MASK
    plt.subplot(1, 3, 2)

    plt.imshow(
        final_mask,
        cmap='gray'
    )

    plt.title("Filtered Mask")

    plt.axis('off')

    # CONTOURS
    plt.subplot(1, 3, 3)

    plt.imshow(
        cv2.cvtColor(
            contour_output,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"Semantic Contours ({detected})"
    )

    plt.axis('off')

    plt.tight_layout()

    plt.show()

    print(f"Processed: {file_name}")