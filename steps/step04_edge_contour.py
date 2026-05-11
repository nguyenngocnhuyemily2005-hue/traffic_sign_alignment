import cv2
import numpy as np


# ===================================================
# STEP 4 — SEMANTIC CONTOUR FILTERING
#
# Input:
# - ROI image
# - Clean binary mask
#
# Output:
# - Detected traffic-sign contours
#
# Goal:
# Detect traffic-sign-like shapes:
# - Triangle
# - Rectangle
# - Circle
# ===================================================


def semantic_contours(mask, roi):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = roi.copy()

    detected = []

    roi_h, roi_w = roi.shape[:2]

    for cnt in contours:

        # ===================================================
        # BASIC AREA FILTER
        # ===================================================

        area = cv2.contourArea(cnt)

        if area < 250:
            continue

        perimeter = cv2.arcLength(
            cnt,
            True
        )

        if perimeter == 0:
            continue

        # ===================================================
        # POLYGON APPROXIMATION
        # Convert complex contour → simple polygon
        # ===================================================

        epsilon = 0.02 * perimeter

        approx = cv2.approxPolyDP(
            cnt,
            epsilon,
            True
        )

        corners = len(approx)

        # ===================================================
        # BOUNDING BOX
        # ===================================================

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / float(h)

        fill_ratio = area / float(w * h)

        center_x = x + w // 2
        center_y = y + h // 2

        # ===================================================
        # CIRCULARITY
        # ===================================================

        circularity = (
            4 * np.pi * area
        ) / (perimeter * perimeter)

        # ===================================================
        # POSITION FILTERING
        # ===================================================

        if center_y > roi_h * 0.90:
            continue

        if center_x < roi_w * 0.10:
            continue

        # ===================================================
        # REMOVE THIN OBJECTS
        # ===================================================

        if aspect_ratio < 0.35:
            continue

        # ===================================================
        # REMOVE EMPTY OBJECTS
        # ===================================================

        if fill_ratio < 0.20:
            continue

        # ===================================================
        # SHAPE CLASSIFICATION
        # ===================================================

        shape_type = None

        color = (255, 255, 255)

        priority = 0

        # ---------------------------------------------------
        # TRIANGLE
        # ---------------------------------------------------

        if corners == 3:

            shape_type = "triangle"

            color = (0, 255, 0)

            priority = 3

        # ---------------------------------------------------
        # RECTANGLE
        # ---------------------------------------------------

        elif corners == 4:

            if 0.5 <= aspect_ratio <= 2.5:

                shape_type = "rectangle"

                color = (255, 0, 0)

                priority = 1

        # ---------------------------------------------------
        # CIRCLE
        # ---------------------------------------------------

        elif circularity > 0.65:

            shape_type = "circle"

            color = (0, 0, 255)

            priority = 2

        # ---------------------------------------------------
        # REJECT UNKNOWN SHAPES
        # ---------------------------------------------------

        else:
            continue

        # ===================================================
        # STORE DETECTION
        # ===================================================

        detected.append({

            'shape': shape_type,

            'priority': priority,

            'area': area,

            'bbox': (x, y, w, h),

            'contour': approx,

            'color': color
        })

    # =======================================================
    # SORT DETECTIONS
    #
    # Priority:
    # triangle > circle > rectangle
    # =======================================================

    detected = sorted(
        detected,
        key=lambda d: (
            d['priority'],
            d['area']
        ),
        reverse=True
    )

    # =======================================================
    # DRAW RESULTS
    # =======================================================

    for obj in detected:

        x, y, w, h = obj['bbox']

        color = obj['color']

        shape_type = obj['shape']

        contour = obj['contour']

        cv2.drawContours(
            output,
            [contour],
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