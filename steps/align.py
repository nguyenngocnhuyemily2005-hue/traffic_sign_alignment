# steps/align.py
# Image alignment step

import cv2
import numpy as np

def align_image(image, reference_image=None):
    """Align an image using feature matching or template matching."""
    if reference_image is None:
        # If no reference, assume the image is already aligned or apply basic alignment
        return image

    # Convert to grayscale
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Use ORB detector
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Warp image
    height, width = reference_image.shape[:2]
    aligned_image = cv2.warpPerspective(image, h, (width, height))

    return aligned_image

def rotate_image(image, angle):
    """Rotate image by a given angle."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))