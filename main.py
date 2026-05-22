import os
import cv2
from matplotlib import pyplot as plt


# ---------------------------------------------------
# IMPORT CONFIG
# ---------------------------------------------------

from config import INPUT_DIR


# ---------------------------------------------------
# IMPORT UTILITIES
# ---------------------------------------------------

from utils.io import (
    create_output_dir,
    load_image,
    save_image
)


# ---------------------------------------------------
# IMPORT PIPELINE STEPS
# ---------------------------------------------------

from steps.step01_roi import extract_roi

from steps.step02_hsv_filter import hsv_filter

from steps.step03_morphology import (
    apply_closing,
    area_filter
)

from steps.step04_contour import (
    extract_contours,
    draw_contours
)


# ---------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ---------------------------------------------------

create_output_dir()


# ---------------------------------------------------
# LOAD INPUT IMAGES
# ---------------------------------------------------

image_files = os.listdir(INPUT_DIR)


# ---------------------------------------------------
# PROCESS PIPELINE
# ---------------------------------------------------

for file_name in image_files:

    image_path = os.path.join(
        INPUT_DIR,
        file_name
    )

    image = load_image(image_path)

    # ------------------------------------------------
    # STEP 1 — ROI
    # ------------------------------------------------

    roi = extract_roi(image)

    # ------------------------------------------------
    # STEP 2 — HSV FILTERING
    # ------------------------------------------------

    blue_mask, red_mask, brightness, scene_type = hsv_filter(roi)

    # ------------------------------------------------
    # STEP 3 — MORPHOLOGY
    # ------------------------------------------------

    blue_closed = apply_closing(blue_mask)

    red_closed = apply_closing(red_mask)

    combined_mask = cv2.add(
        blue_closed,
        red_closed
    )

    filtered_mask, contours = area_filter(
        combined_mask
    )

    # ------------------------------------------------
    # STEP 4 — CONTOUR EXTRACTION
    # ------------------------------------------------

    final_contours = extract_contours(
        filtered_mask
    )

    contour_visualization = draw_contours(
        roi,
        final_contours
    )

    # ------------------------------------------------
    # SAVE OUTPUT
    # ------------------------------------------------

    save_image(
        f'contours_{file_name}',
        contour_visualization
    )

    # ------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------

    plt.figure(figsize=(16, 5))

    # ROI
    plt.subplot(1, 3, 1)

    plt.imshow(
        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    )

    plt.title(f'ROI ({scene_type})')

    plt.axis('off')

    # Filtered Mask
    plt.subplot(1, 3, 2)

    plt.imshow(
        filtered_mask,
        cmap='gray'
    )

    plt.title('Filtered Mask')

    plt.axis('off')

    # Contours
    plt.subplot(1, 3, 3)

    plt.imshow(
        cv2.cvtColor(
            contour_visualization,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f'Contours ({len(final_contours)})'
    )

    plt.axis('off')

    plt.suptitle(
        f'{file_name} | Brightness = {brightness:.2f}',
        fontsize=14
    )

    plt.tight_layout()

    plt.show()

    print(f"Processed: {file_name}")