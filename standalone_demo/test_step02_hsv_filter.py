import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


# ---------------------------------------------------
# STEP 2 — ADAPTIVE HSV FILTERING
# Using:
# - Day / Night brightness estimation
# - Adaptive HSV thresholds
# - Separate blue / red masks
# ---------------------------------------------------

def hsv_filter(roi):

    # ------------------------------------------------
    # Blur to stabilize HSV
    # ------------------------------------------------

    blurred = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    # ------------------------------------------------
    # Estimate brightness
    # ------------------------------------------------

    gray = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    # ------------------------------------------------
    # Convert BGR → HSV
    # ------------------------------------------------

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # ------------------------------------------------
    # DAY / NIGHT DECISION
    # ------------------------------------------------

    if brightness < 130:

        scene_type = "NIGHT"

        # --------------------------------------------
        # NIGHTTIME BLUE THRESHOLDS
        # Lower S and V to recover weak blue regions
        # --------------------------------------------

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([135, 255, 255])

        # --------------------------------------------
        # NIGHTTIME RED THRESHOLDS
        # Slightly tighter to reduce reflections
        # --------------------------------------------

        lower_red1 = np.array([0, 160, 70])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 160, 70])
        upper_red2 = np.array([180, 255, 255])

    else:

        scene_type = "DAY"

        # --------------------------------------------
        # DAYTIME BLUE THRESHOLDS
        # --------------------------------------------

        lower_blue = np.array([100, 120, 70])
        upper_blue = np.array([130, 255, 255])

        # --------------------------------------------
        # DAYTIME RED THRESHOLDS
        # --------------------------------------------

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

    # ------------------------------------------------
    # PRINT DEBUG INFO
    # ------------------------------------------------

    print("\n-----------------------------------")
    print(f"Scene Type : {scene_type}")
    print(f"Brightness : {brightness:.2f}")

    print("\nBlue Thresholds:")
    print(f"Lower : {lower_blue}")
    print(f"Upper : {upper_blue}")

    print("\nRed Thresholds:")
    print(f"Lower Red 1 : {lower_red1}")
    print(f"Upper Red 1 : {upper_red1}")
    print(f"Lower Red 2 : {lower_red2}")
    print(f"Upper Red 2 : {upper_red2}")

    print("-----------------------------------")

    return blue_mask, red_mask, brightness, scene_type


# ---------------------------------------------------
# SIMPLE ROI FUNCTION
# ---------------------------------------------------

def extract_roi(image):

    h, w = image.shape[:2]

    # Keep right-half region
    x_start = int(w * 0.50)

    # Keep most upper area
    y_start = 0
    y_end = int(h * 0.85)

    roi = image[
        y_start:y_end,
        x_start:w
    ]

    return roi


# ---------------------------------------------------
# CREATE OUTPUT FOLDER
# ---------------------------------------------------

os.makedirs('output_images', exist_ok=True)

input_folder = 'input_images'

image_files = os.listdir(input_folder)


# ---------------------------------------------------
# PROCESS ALL IMAGES
# ---------------------------------------------------

for file_name in image_files:

    image_path = os.path.join(
        input_folder,
        file_name
    )

    img = cv2.imread(image_path)

    if img is None:

        print(f"Cannot load image: {file_name}")
        continue

    # ---------------------------------------------
    # STEP 1 — ROI
    # ---------------------------------------------

    roi = extract_roi(img)

    # ---------------------------------------------
    # STEP 2 — HSV FILTERING
    # ---------------------------------------------

    blue_mask, red_mask, brightness, scene_type = hsv_filter(roi)

    # ---------------------------------------------
    # SAVE OUTPUTS
    # ---------------------------------------------

    cv2.imwrite(
        f'output_images/blue_mask_{file_name}',
        blue_mask
    )

    cv2.imwrite(
        f'output_images/red_mask_{file_name}',
        red_mask
    )

    # ---------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------

    titles = [
        f'ROI ({scene_type})',
        'Blue Mask',
        'Red Mask',
    ]

    images = [
        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB),
        blue_mask,
        red_mask,
    ]

    plt.figure(figsize=(16, 5))

    for i in range(3):

        plt.subplot(1, 3, i + 1)

        if i == 0:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap='gray')

        plt.title(titles[i])

        plt.axis('off')

    plt.suptitle(
        f'{file_name} | Brightness = {brightness:.2f}',
        fontsize=14
    )

    plt.tight_layout()

    plt.show()

    print(f"\nProcessed: {file_name}")