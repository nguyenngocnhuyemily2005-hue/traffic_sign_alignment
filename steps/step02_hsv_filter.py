import cv2
import numpy as np


# ---------------------------------------------------
# STEP 2 — HSV FILTERING
# ---------------------------------------------------

def hsv_filter(roi):

    # ------------------------------------------------
    # GAUSSIAN BLUR
    # ------------------------------------------------

    blurred = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    # ------------------------------------------------
    # BRIGHTNESS ESTIMATION
    # ------------------------------------------------

    gray = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    # ------------------------------------------------
    # HSV CONVERSION
    # ------------------------------------------------

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV
    )

    # ------------------------------------------------
    # DAY / NIGHT ADAPTIVE THRESHOLDS
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

    return (
        blue_mask,
        red_mask,
        brightness,
        scene_type
    )