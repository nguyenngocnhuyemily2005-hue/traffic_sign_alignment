import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


# ---------------------------------------------------
# STEP 3 — AREA FILTERING EXPERIMENT
#
# Compare:
# - min_area = 50
# - min_area = 100
# - min_area = 200
#
# Using:
# - Adaptive HSV
# - Morphology Closing (5x5)
# - Separate Blue / Red Morphology
# ---------------------------------------------------


# ---------------------------------------------------
# ROI EXTRACTION
# ---------------------------------------------------

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


# ---------------------------------------------------
# HSV FILTERING
# ---------------------------------------------------

def hsv_filter(roi):

    blurred = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    gray = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # ------------------------------------------------
    # DAY / NIGHT SWITCH
    # ------------------------------------------------

    if brightness < 130:

        scene_type = "NIGHT"

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([135, 255, 255])

        lower_red1 = np.array([0, 160, 70])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 160, 70])
        upper_red2 = np.array([180, 255, 255])

    else:

        scene_type = "DAY"

        lower_blue = np.array([100, 120, 70])
        upper_blue = np.array([130, 255, 255])

        lower_red1 = np.array([0, 140, 80])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 140, 80])
        upper_red2 = np.array([180, 255, 255])

    # ------------------------------------------------
    # BLUE MASK
    # ------------------------------------------------

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    # ------------------------------------------------
    # RED MASK
    # ------------------------------------------------

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

    return blue_mask, red_mask, brightness, scene_type


# ---------------------------------------------------
# MORPHOLOGY CLOSING
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

def area_filter(mask, min_area):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered = np.zeros_like(mask)

    kept_contours = 0

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

            kept_contours += 1

    return filtered, kept_contours


# ---------------------------------------------------
# PROCESS IMAGES
# ---------------------------------------------------

input_folder = 'input_images'

image_files = os.listdir(input_folder)

for file_name in image_files:

    image_path = os.path.join(
        input_folder,
        file_name
    )

    img = cv2.imread(image_path)

    if img is None:

        print(f"Cannot load image: {file_name}")
        continue

    # ------------------------------------------------
    # ROI
    # ------------------------------------------------

    roi = extract_roi(img)

    # ------------------------------------------------
    # HSV
    # ------------------------------------------------

    blue_mask, red_mask, brightness, scene_type = hsv_filter(roi)

    # ------------------------------------------------
    # MORPHOLOGY
    # ------------------------------------------------

    blue_closed = apply_closing(blue_mask)

    red_closed = apply_closing(red_mask)

    combined = cv2.add(
        blue_closed,
        red_closed
    )

    # ------------------------------------------------
    # TEST AREA THRESHOLDS
    # ------------------------------------------------

    thresholds = [50, 100, 200]

    results = []

    contour_counts = []

    for threshold in thresholds:

        filtered, kept = area_filter(
            combined,
            threshold
        )

        results.append(filtered)

        contour_counts.append(kept)

        print("\n-----------------------------------")
        print(f"File              : {file_name}")
        print(f"Scene Type        : {scene_type}")
        print(f"Brightness        : {brightness:.2f}")
        print(f"Min Area          : {threshold}")
        print(f"Contours Kept     : {kept}")
        print("-----------------------------------")

    # ------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------

    plt.figure(figsize=(20, 5))

    # ROI
    plt.subplot(1, 4, 1)

    plt.imshow(
        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    )

    plt.title(f'ROI ({scene_type})')

    plt.axis('off')

    # Threshold 50
    plt.subplot(1, 4, 2)

    plt.imshow(
        results[0],
        cmap='gray'
    )

    plt.title(
        f'Area ≥ 50\nContours: {contour_counts[0]}'
    )

    plt.axis('off')

    # Threshold 100
    plt.subplot(1, 4, 3)

    plt.imshow(
        results[1],
        cmap='gray'
    )

    plt.title(
        f'Area ≥ 100\nContours: {contour_counts[1]}'
    )

    plt.axis('off')

    # Threshold 200
    plt.subplot(1, 4, 4)

    plt.imshow(
        results[2],
        cmap='gray'
    )

    plt.title(
        f'Area ≥ 200\nContours: {contour_counts[2]}'
    )

    plt.axis('off')

    plt.suptitle(
        f'{file_name} | Brightness = {brightness:.2f}',
        fontsize=14
    )

    plt.tight_layout()

    plt.show()