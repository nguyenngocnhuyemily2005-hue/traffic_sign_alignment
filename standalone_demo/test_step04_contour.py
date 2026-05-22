import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


# ---------------------------------------------------
# STEP 4 — CONTOUR COUNT COMPARISON
#
# Compare contour counts across:
# - Raw HSV mask
# - After morphology
# - After area filtering
#
# Goal:
# Show how preprocessing stabilizes
# contour extraction progressively.
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
# MORPHOLOGY
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


# ---------------------------------------------------
# COUNT CONTOURS
# ---------------------------------------------------

def count_contours(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return len(contours)


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

    raw_combined = cv2.add(
        blue_mask,
        red_mask
    )

    # ------------------------------------------------
    # CONTOUR COUNT — RAW HSV
    # ------------------------------------------------

    raw_count = count_contours(raw_combined)

    # ------------------------------------------------
    # MORPHOLOGY
    # ------------------------------------------------

    blue_closed = apply_closing(blue_mask)

    red_closed = apply_closing(red_mask)

    morphology_combined = cv2.add(
        blue_closed,
        red_closed
    )

    # ------------------------------------------------
    # CONTOUR COUNT — MORPHOLOGY
    # ------------------------------------------------

    morphology_count = count_contours(
        morphology_combined
    )

    # ------------------------------------------------
    # AREA FILTERING
    # ------------------------------------------------

    filtered_mask, final_contours = area_filter(
        morphology_combined,
        min_area=100
    )

    # ------------------------------------------------
    # CONTOUR COUNT — FINAL
    # ------------------------------------------------

    final_count = len(final_contours)

    # ------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------

    print("\n===================================")
    print(f"File           : {file_name}")
    print(f"Scene Type     : {scene_type}")
    print(f"Brightness     : {brightness:.2f}")
    print("===================================")

    print(f"Raw HSV Contours          : {raw_count}")
    print(f"After Morphology          : {morphology_count}")
    print(f"After Area Filtering      : {final_count}")

    print("===================================\n")

    # ------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------

    stages = [
        raw_combined,
        morphology_combined,
        filtered_mask
    ]

    titles = [
        f'Raw HSV\nContours: {raw_count}',
        f'After Morphology\nContours: {morphology_count}',
        f'After Area Filtering\nContours: {final_count}'
    ]

    plt.figure(figsize=(18, 5))

    # ROI
    plt.subplot(1, 4, 1)

    plt.imshow(
        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    )

    plt.title(f'ROI ({scene_type})')

    plt.axis('off')

    # Processing stages
    for i in range(3):

        plt.subplot(1, 4, i + 2)

        plt.imshow(
            stages[i],
            cmap='gray'
        )

        plt.title(titles[i])

        plt.axis('off')

    plt.suptitle(
        f'{file_name} | Brightness = {brightness:.2f}',
        fontsize=14
    )

    plt.tight_layout()

    plt.show()