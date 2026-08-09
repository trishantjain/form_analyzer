"""Reference-template layout matching.

The matcher is intentionally a *secondary* signal. It uses ORB features and a
homography to tolerate moderate scale, rotation and camera perspective. It
must not be used as a pixel-perfect comparison because filled values and
handwriting legitimately vary between valid CSR forms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class TemplateMatchResult:
    matched: bool
    similarity: float
    reference_name: Optional[str]
    good_matches: int
    inlier_ratio: float


def _score(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, int, float]:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    cand_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY) if candidate.ndim == 3 else candidate

    # Reduce the influence of handwriting/filled values and keep structural edges.
    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    cand_gray = cv2.GaussianBlur(cand_gray, (5, 5), 0)
    ref_edges = cv2.Canny(ref_gray, 60, 160)
    cand_edges = cv2.Canny(cand_gray, 60, 160)

    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=12)
    kp1, des1 = orb.detectAndCompute(ref_edges, None)
    kp2, des2 = orb.detectAndCompute(cand_edges, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0.0, 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.78 * n.distance]
    if len(good) < 8:
        return min(len(good) / 20.0, 0.25), len(good), 0.0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if H is None or mask is None:
        return 0.0, len(good), 0.0
    inlier_ratio = float(mask.sum()) / len(mask)
    match_strength = min(len(good) / 60.0, 1.0)
    similarity = 0.55 * inlier_ratio + 0.45 * match_strength
    return float(similarity), len(good), float(inlier_ratio)


def compare_to_references(
    candidate: np.ndarray,
    references: List[tuple[str, np.ndarray]],
    threshold: float = 0.30,
) -> TemplateMatchResult:
    best = TemplateMatchResult(False, 0.0, None, 0, 0.0)
    for name, ref in references:
        try:
            similarity, good, inliers = _score(ref, candidate)
        except Exception:
            continue
        if similarity > best.similarity:
            best = TemplateMatchResult(
                matched=similarity >= threshold,
                similarity=round(similarity, 4),
                reference_name=name,
                good_matches=good,
                inlier_ratio=round(inliers, 4),
            )
    return best
