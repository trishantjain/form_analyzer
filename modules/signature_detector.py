"""Signature area analysis.

IMPORTANT: This module only checks whether ink-like marks exist within a
signature region. It does NOT verify whose signature it is, whether it
matches a specimen, or whether it is genuine/forged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from modules.ocr_engine import BBox


@dataclass
class SignatureDetectionResult:
    present: bool
    bbox: Optional[BBox]
    ink_percentage: float
    warning: Optional[str]


def analyze_signature_region(
    page_gray: np.ndarray,
    region_bbox: BBox,
    min_ink_percentage: float = 1.5,
) -> SignatureDetectionResult:
    """Analyze a (configured or detected) region for ink-like marks.

    Args:
        page_gray: Grayscale page image.
        region_bbox: Bounding box (x_min, y_min, x_max, y_max) of the area
            to check, e.g. a "Signature" field box either configured
            manually or located near a matched "Signature" heading.
        min_ink_percentage: Minimum percentage of dark pixels within the
            region required to consider a signature present.

    Returns:
        SignatureDetectionResult with ink coverage percentage and a
        present/missing determination.
    """
    x_min, y_min, x_max, y_max = region_bbox
    height, width = page_gray.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(width, x_max), min(height, y_max)

    if x_max <= x_min or y_max <= y_min:
        return SignatureDetectionResult(
            present=False,
            bbox=region_bbox,
            ink_percentage=0.0,
            warning="Signature region coordinates are invalid or out of bounds.",
        )

    crop = page_gray[y_min:y_max, x_min:x_max]

    # Otsu thresholding to separate ink (dark) from paper (light).
    _thresh_val, binary = cv2.threshold(
        crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    dark_pixel_count = int(np.count_nonzero(binary))
    total_pixels = binary.size
    ink_percentage = (dark_pixel_count / total_pixels) * 100.0 if total_pixels else 0.0

    # Count contours to distinguish scribble-like ink from a solid smudge
    # or scanning artifact (a genuine signature typically has multiple
    # small disconnected strokes).
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stroke_like_contours = [c for c in contours if cv2.contourArea(c) > 3]

    present = ink_percentage >= min_ink_percentage and len(stroke_like_contours) >= 2

    warning = None
    if not present:
        if ink_percentage < min_ink_percentage:
            warning = "Signature region has low ink coverage; it may be blank."
        else:
            warning = (
                "Signature region has ink but does not resemble handwritten "
                "strokes (possible smudge or scan artifact)."
            )

    return SignatureDetectionResult(
        present=present,
        bbox=(x_min, y_min, x_max, y_max),
        ink_percentage=round(ink_percentage, 2),
        warning=warning,
    )
