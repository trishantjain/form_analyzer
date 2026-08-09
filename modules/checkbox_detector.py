"""Checkbox detection using OpenCV contour analysis.

Detects small square/rectangular outlined boxes, determines whether each is
filled ("checked") based on interior dark-pixel density, and attempts to
associate each box with the nearest OCR text as its label. If automatic
label matching is unreliable, the caller can supply manual coordinates
instead (see `label_checkbox_by_position`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from modules.ocr_engine import BBox, OCRTextBlock

MIN_BOX_SIZE = 12
MAX_BOX_SIZE = 60
SQUARENESS_TOLERANCE = 0.35  # allowed relative difference between w and h


@dataclass
class CheckboxDetection:
    label: Optional[str]
    checked: bool
    bbox: BBox
    confidence: float


def detect_checkboxes(
    page_gray: np.ndarray,
    ocr_blocks: Optional[List[OCRTextBlock]] = None,
    fill_threshold: float = 0.25,
) -> List[CheckboxDetection]:
    """Detect checkbox-like squares on a page and their checked state.

    Args:
        page_gray: Grayscale page image.
        ocr_blocks: OCR text blocks for the same page, used to assign a
            nearby label to each detected checkbox (best-effort).
        fill_threshold: Fraction of dark pixels inside a box above which it
            is considered "checked".

    Returns:
        List of CheckboxDetection, left-to-right, top-to-bottom.
    """
    binary = cv2.adaptiveThreshold(
        page_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=8,
    )

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    detections: List[CheckboxDetection] = []
    if hierarchy is None:
        return detections

    hierarchy = hierarchy[0]

    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        if not (MIN_BOX_SIZE <= w <= MAX_BOX_SIZE and MIN_BOX_SIZE <= h <= MAX_BOX_SIZE):
            continue

        if abs(w - h) / max(w, h) > SQUARENESS_TOLERANCE:
            continue

        # Approximate the contour polygon; a checkbox outline should be
        # roughly quadrilateral.
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        interior = page_gray[y : y + h, x : x + w]
        if interior.size == 0:
            continue

        _thresh_val, interior_binary = cv2.threshold(
            interior, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        # Ignore the border pixels (the box outline itself) when measuring
        # fill, by looking only at the inner 60% of the box.
        margin_x, margin_y = max(1, int(w * 0.2)), max(1, int(h * 0.2))
        inner = interior_binary[margin_y : h - margin_y, margin_x : w - margin_x]
        fill_ratio = float(np.count_nonzero(inner)) / inner.size if inner.size else 0.0

        checked = fill_ratio >= fill_threshold
        confidence = min(1.0, 0.5 + fill_ratio) if checked else max(0.3, 1.0 - fill_ratio)

        bbox = (x, y, x + w, y + h)
        label = _find_nearest_label(bbox, ocr_blocks) if ocr_blocks else None

        detections.append(
            CheckboxDetection(
                label=label,
                checked=checked,
                bbox=bbox,
                confidence=round(confidence, 3),
            )
        )

    detections.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
    return detections


def _find_nearest_label(
    box_bbox: BBox, ocr_blocks: List[OCRTextBlock], max_distance: int = 150
) -> Optional[str]:
    """Find the OCR text block closest to (typically to the right of) a checkbox."""
    x_min, y_min, x_max, y_max = box_bbox
    box_center_y = (y_min + y_max) / 2

    best_text = None
    best_distance = max_distance

    for block in ocr_blocks:
        bx_min, by_min, bx_max, by_max = block.bbox
        block_center_y = (by_min + by_max) / 2

        # Only consider text roughly to the right of the box and vertically
        # aligned with it (typical "[ ] Label" layout).
        if bx_min < x_max - 5:
            continue
        if abs(block_center_y - box_center_y) > 20:
            continue

        distance = bx_min - x_max
        if 0 <= distance < best_distance:
            best_distance = distance
            best_text = block.text

    return best_text


def label_checkbox_by_position(
    detections: List[CheckboxDetection], manual_labels: List[dict]
) -> List[CheckboxDetection]:
    """Override/assign labels for checkboxes using user-supplied manual
    coordinates when automatic label matching is unreliable.

    Args:
        detections: Output of detect_checkboxes().
        manual_labels: List of dicts like
            {"label": "Male", "bbox": [x1, y1, x2, y2]}.

    Returns:
        Updated list of CheckboxDetection with labels overridden where a
        manual bbox overlaps a detected checkbox.
    """
    def _iou(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter_area / float(area_a + area_b - inter_area)

    for manual in manual_labels:
        manual_bbox = tuple(manual["bbox"])
        for det in detections:
            if _iou(det.bbox, manual_bbox) > 0.3:
                det.label = manual["label"]

    return detections
