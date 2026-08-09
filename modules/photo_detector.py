"""Student photograph region detection.

This module does NOT perform face recognition or identity verification.
It only checks whether a photograph-like rectangular region exists on the
page, based on its size, aspect ratio, and whether it is "blank" (has
enough visual variance/content to plausibly be a photo rather than empty
whitespace or a printed box outline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from modules.layout_analysis import LayoutRegion
from modules.ocr_engine import BBox


@dataclass
class PhotoDetectionResult:
    found: bool
    bbox: Optional[BBox]
    width: int
    height: int
    aspect_ratio: float
    quality_warning: Optional[str]


def _region_is_blank(gray_region: np.ndarray, std_threshold: float = 12.0) -> bool:
    """Heuristic: a nearly blank region (plain background or box outline
    only) has very low pixel-intensity standard deviation.
    """
    if gray_region.size == 0:
        return True
    return float(np.std(gray_region)) < std_threshold


def detect_photo_region(
    page_gray: np.ndarray,
    candidate_regions: List[LayoutRegion],
    min_width: int = 100,
    min_height: int = 100,
    aspect_ratio_min: float = 0.6,
    aspect_ratio_max: float = 1.2,
) -> PhotoDetectionResult:
    """Pick the best candidate non-text region that looks like a photo slot.

    Args:
        page_gray: Grayscale page image.
        candidate_regions: Non-text regions from layout_analysis, typically
            already sorted by area descending.
        min_width: Minimum acceptable photo width in pixels.
        min_height: Minimum acceptable photo height in pixels.
        aspect_ratio_min: Minimum acceptable width/height ratio.
        aspect_ratio_max: Maximum acceptable width/height ratio.

    Returns:
        PhotoDetectionResult describing the best match, or found=False if
        no candidate region satisfies the size/aspect-ratio constraints.
    """
    best_region: Optional[LayoutRegion] = None
    best_region_blank = True

    for region in candidate_regions:
        if region.width < min_width or region.height < min_height:
            continue

        aspect_ratio = region.width / float(region.height)
        if not (aspect_ratio_min <= aspect_ratio <= aspect_ratio_max):
            continue

        x_min, y_min, x_max, y_max = region.bbox
        crop = page_gray[y_min:y_max, x_min:x_max]
        is_blank = _region_is_blank(crop)

        # Prefer the largest region that is NOT blank (i.e., actually has
        # photographic content rather than being an empty printed box).
        if not is_blank:
            best_region = region
            best_region_blank = False
            break

        if best_region is None:
            best_region = region
            best_region_blank = True

    if best_region is None:
        return PhotoDetectionResult(
            found=False,
            bbox=None,
            width=0,
            height=0,
            aspect_ratio=0.0,
            quality_warning="No region matching expected photo size/aspect ratio was found.",
        )

    aspect_ratio = best_region.width / float(best_region.height)
    warning = None
    if best_region_blank:
        warning = (
            "A photo-sized region was found but appears blank or empty "
            "(low visual detail)."
        )
    elif best_region.width < min_width * 1.3 or best_region.height < min_height * 1.3:
        warning = "Detected photo region is close to the minimum allowed size."

    return PhotoDetectionResult(
        found=not best_region_blank,
        bbox=best_region.bbox,
        width=best_region.width,
        height=best_region.height,
        aspect_ratio=round(aspect_ratio, 3),
        quality_warning=warning,
    )
