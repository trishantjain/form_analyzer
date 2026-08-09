"""Image preprocessing utilities: rotation correction, resizing, thresholding.

All functions accept and return numpy arrays in BGR format (OpenCV
convention) unless otherwise noted, so they can be chained directly with
OpenCV-based detectors.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MAX_DIMENSION = 2200  # Longest side, in pixels, after resizing.


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR numpy array."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR numpy array to a PIL RGB image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_for_processing(
    image: np.ndarray, max_dimension: int = MAX_DIMENSION
) -> Tuple[np.ndarray, float]:
    """Resize an image so its longest side does not exceed max_dimension.

    Args:
        image: OpenCV BGR image.
        max_dimension: Maximum allowed length of the longer side.

    Returns:
        Tuple of (resized_image, scale_factor). scale_factor is
        resized_size / original_size, useful for mapping bounding boxes
        back to the original resolution if needed.
    """
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dimension:
        return image, 1.0

    scale = max_dimension / float(longest_side)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def correct_rotation(image: np.ndarray) -> np.ndarray:
    """Attempt to correct small skew/rotation using minAreaRect on text-like pixels.

    This is a lightweight geometric deskew (not a full OSD/orientation
    classifier). It is best-effort: if it cannot confidently determine a
    rotation angle, it returns the original image unchanged.

    Args:
        image: OpenCV BGR image.

    Returns:
        Deskewed OpenCV BGR image (or the original if deskew was not
        confidently possible).
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]

        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 100:
            # Not enough foreground pixels to reliably estimate an angle.
            return image

        angle = cv2.minAreaRect(coords)[-1]

        # cv2.minAreaRect returns an angle in [-90, 0); normalize it.
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only correct small skews; large "rotations" from minAreaRect on
        # dense text blocks are usually noise, not a genuinely rotated page.
        if abs(angle) < 0.5 or abs(angle) > 15:
            return image

        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
    except Exception:  # noqa: BLE001
        logger.exception("Rotation correction failed; returning original image.")
        return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adaptive_threshold(gray_image: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding + light denoising to a grayscale image."""
    denoised = cv2.fastNlMeansDenoising(gray_image, h=10)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )
    return thresholded


def preprocess_page(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Run the standard preprocessing pipeline on a page image.

    Args:
        image: Original PIL page image (kept elsewhere, untouched, for
            display purposes).

    Returns:
        Tuple of (display_image, processed_image):
            display_image: resized + deskewed BGR image, suitable for
                annotation and display (still color).
            processed_image: grayscale + thresholded version, useful for
                some detectors that benefit from binarized input.
    """
    cv_image = pil_to_cv2(image)
    resized, _scale = resize_for_processing(cv_image)
    deskewed = correct_rotation(resized)
    gray = to_grayscale(deskewed)
    processed = adaptive_threshold(gray)
    return deskewed, processed
