"""Layout analysis helpers.

Provides two things:
  1. Grouping of raw OCR text blocks into logical lines (useful for
     matching multi-word headings that OCR may split across boxes).
  2. Detection of "non-text" rectangular regions in the page — these are
     candidate photo / stamp / logo regions, found by looking at large
     areas of the page that contain no OCR text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from modules.ocr_engine import BBox, OCRTextBlock


@dataclass
class LayoutRegion:
    """A rectangular region of the page with no recognized text inside it."""

    bbox: BBox
    width: int
    height: int
    area: int


def group_text_lines(blocks: List[OCRTextBlock], y_tolerance: int = 12) -> List[str]:
    """Group OCR blocks that share a similar vertical position into lines.

    This helps fuzzy-heading matching succeed even when OCR splits a
    heading like "Student Name" into two separate boxes ("Student", "Name").

    Args:
        blocks: List of OCRTextBlock from OCREngine.run().
        y_tolerance: Maximum vertical pixel difference for two blocks to be
            considered part of the same line.

    Returns:
        List of joined line strings, ordered top-to-bottom, left-to-right.
    """
    if not blocks:
        return []

    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    lines: List[List[OCRTextBlock]] = []

    for block in sorted_blocks:
        placed = False
        block_center_y = (block.bbox[1] + block.bbox[3]) / 2
        for line in lines:
            line_center_y = (line[0].bbox[1] + line[0].bbox[3]) / 2
            if abs(block_center_y - line_center_y) <= y_tolerance:
                line.append(block)
                placed = True
                break
        if not placed:
            lines.append([block])

    joined_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda b: b.bbox[0])
        joined_lines.append(" ".join(b.text for b in line_sorted))

    return joined_lines


def find_non_text_regions(
    image_shape: Tuple[int, int],
    text_blocks: List[OCRTextBlock],
    min_width: int = 60,
    min_height: int = 60,
    page_gray: "np.ndarray | None" = None,
    text_overlap_threshold: float = 0.35,
) -> List[LayoutRegion]:
    """Find candidate non-text rectangular content regions (photos, stamps,
    logos) using edge-density blob detection.

    Rationale: simply inverting a mask of OCR text boxes tends to leave one
    single, huge "blank page" contour (everything that isn't text), which is
    not useful on sparse forms. Instead, this looks for blobs of visual
    *content* (detected via edges), then discards any blob that overlaps
    heavily with a known OCR text box -- what's left are non-text content
    regions such as photos, stamps, and logos.

    Args:
        image_shape: (height, width) of the page.
        text_blocks: OCR text blocks already detected on this page.
        min_width: Minimum region width to keep.
        min_height: Minimum region height to keep.
        page_gray: Grayscale page image, required to compute edge density.
            If not supplied, an empty list is returned (no image to analyze).
        text_overlap_threshold: If more than this fraction of a candidate
            blob's area overlaps with OCR text boxes, it is discarded as
            text rather than kept as a photo/stamp/logo candidate.

    Returns:
        List of LayoutRegion sorted by area, descending.
    """
    if page_gray is None:
        return []

    height, width = image_shape[:2]

    text_mask = np.zeros((height, width), dtype=np.uint8)
    for block in text_blocks:
        x_min, y_min, x_max, y_max = block.bbox
        cv2.rectangle(
            text_mask,
            (max(0, x_min), max(0, y_min)),
            (min(width, x_max), min(height, y_max)),
            255,
            thickness=-1,
        )

    edges = cv2.Canny(page_gray, 50, 150)
    # Dilate to merge nearby edges (e.g. photo texture, stamp outline) into
    # single connected blobs rather than many tiny fragments.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[LayoutRegion] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_width or h < min_height:
            continue
        if w > width * 0.98 and h > height * 0.98:
            continue  # whole-page artifact, not a discrete element

        blob_area = w * h
        overlap_area = int(np.count_nonzero(text_mask[y : y + h, x : x + w]))
        overlap_fraction = overlap_area / blob_area if blob_area else 1.0

        if overlap_fraction > text_overlap_threshold:
            continue  # mostly text, not a photo/stamp/logo candidate

        regions.append(LayoutRegion(bbox=(x, y, x + w, y + h), width=w, height=h, area=blob_area))

    regions.sort(key=lambda r: r.area, reverse=True)
    return regions
