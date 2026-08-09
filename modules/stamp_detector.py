"""Stamp / logo detection using OpenCV template matching, with an ORB
feature-matching fallback for cases involving rotation or scale changes.

IMPORTANT: This module verifies *visual similarity* to a reference image.
It does NOT and CANNOT verify authenticity of a stamp, seal, or logo. A
photocopied, scanned, or forged stamp with a similar visual appearance may
still be reported as a match. Always present results with that caveat in
the UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from modules.ocr_engine import BBox

logger = logging.getLogger(__name__)


@dataclass
class StampDetectionResult:
    found: bool
    similarity: float
    bbox: Optional[BBox]
    method: str  # "template_matching" or "orb_feature_matching"


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _template_match_multiscale(
    page_gray: np.ndarray, template_gray: np.ndarray, scales
) -> tuple[float, Optional[BBox]]:
    """Slide the template over the page at multiple scales; return best match."""
    best_score = -1.0
    best_bbox: Optional[BBox] = None

    page_h, page_w = page_gray.shape[:2]

    for scale in scales:
        resized_template = cv2.resize(
            template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        th, tw = resized_template.shape[:2]
        if th < 10 or tw < 10 or th > page_h or tw > page_w:
            continue

        result = cv2.matchTemplate(page_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            x, y = max_loc
            best_bbox = (x, y, x + tw, y + th)

    return best_score, best_bbox


def _orb_match(
    page_gray: np.ndarray, template_gray: np.ndarray, min_matches: int = 10
) -> tuple[float, Optional[BBox]]:
    """Fallback matcher robust to rotation and moderate scale changes."""
    orb = cv2.ORB_create(nfeatures=1500)

    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(page_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0.0, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < min_matches:
        similarity = len(good_matches) / max(min_matches, 1) * 0.5
        return min(similarity, 0.5), None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None:
        return 0.0, None

    inlier_ratio = float(mask.sum()) / len(mask) if mask is not None else 0.0

    th, tw = template_gray.shape[:2]
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    try:
        transformed_corners = cv2.perspectiveTransform(corners, homography)
    except cv2.error:
        return 0.0, None

    xs = transformed_corners[:, 0, 0]
    ys = transformed_corners[:, 0, 1]
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    # Combine match-count confidence with geometric inlier ratio.
    match_score = min(len(good_matches) / 40.0, 1.0)
    similarity = 0.5 * match_score + 0.5 * inlier_ratio

    return similarity, bbox


def detect_stamp(
    page_image: np.ndarray,
    reference_image: np.ndarray,
    similarity_threshold: float = 0.70,
) -> StampDetectionResult:
    """Detect whether a reference stamp/logo appears on a page.

    Args:
        page_image: Full page, OpenCV BGR or grayscale.
        reference_image: Reference stamp/logo image, OpenCV BGR or grayscale.
        similarity_threshold: Minimum similarity (0-1) to count as "found".

    Returns:
        StampDetectionResult with the best result across both methods.
    """
    page_gray = _to_gray(page_image)
    template_gray = _to_gray(reference_image)

    scales = [0.5, 0.65, 0.8, 1.0, 1.2, 1.5]
    tm_score, tm_bbox = _template_match_multiscale(page_gray, template_gray, scales)

    best_score = tm_score
    best_bbox = tm_bbox
    best_method = "template_matching"

    # Use ORB as a fallback (or to improve on) whenever template matching is
    # inconclusive, since rotation/scale hurts template matching badly.
    if tm_score < similarity_threshold:
        try:
            orb_score, orb_bbox = _orb_match(page_gray, template_gray)
        except Exception:  # noqa: BLE001
            logger.exception("ORB fallback matching failed.")
            orb_score, orb_bbox = 0.0, None

        if orb_score > best_score:
            best_score = orb_score
            best_bbox = orb_bbox
            best_method = "orb_feature_matching"

    found = best_score >= similarity_threshold and best_bbox is not None

    return StampDetectionResult(
        found=found,
        similarity=round(float(best_score), 4),
        bbox=best_bbox if found else None,
        method=best_method,
    )
