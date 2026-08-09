"""Generic validator for every saved form template."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

import cv2
import numpy as np
from rapidfuzz import fuzz

from modules.ocr_engine import BBox, OCRTextBlock


@dataclass
class GenericCheck:
    id: str
    label: str
    check_type: str
    required: bool
    passed: bool
    confidence: float = 0.0
    matched_text: Optional[str] = None
    bbox: Optional[BBox] = None
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenericValidation:
    document_type: str
    classification_confidence: float
    checks: List[GenericCheck]
    structure_score: float
    visual_score: float
    overall_score: float
    missing_items: List[str]
    warnings: List[str]
    matched_anchors: List[str] = field(default_factory=list)
    missing_anchors: List[str] = field(default_factory=list)
    geometry_passed: bool = True
    geometry_details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "classification_confidence": self.classification_confidence,
            "checks": [c.to_dict() for c in self.checks],
            "structure_score": self.structure_score,
            "visual_score": self.visual_score,
            "overall_score": self.overall_score,
            "missing_items": self.missing_items,
            "warnings": self.warnings,
            "matched_anchors": self.matched_anchors,
            "missing_anchors": self.missing_anchors,
            "geometry_passed": self.geometry_passed,
            "geometry_details": self.geometry_details,
        }


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_candidates(blocks: Sequence[OCRTextBlock]) -> List[Tuple[str, BBox]]:
    valid = [b for b in blocks if b.text and b.text.strip()]
    candidates = [(b.text, b.bbox) for b in valid]

    sorted_blocks = sorted(
        valid,
        key=lambda b: ((b.bbox[1] + b.bbox[3]) / 2, b.bbox[0]),
    )

    lines: List[List[OCRTextBlock]] = []

    for block in sorted_blocks:
        cy = (block.bbox[1] + block.bbox[3]) / 2
        placed = False

        for line in lines:
            line_cy = sum(
                (b.bbox[1] + b.bbox[3]) / 2 for b in line
            ) / len(line)

            if abs(cy - line_cy) <= 14:
                line.append(block)
                placed = True
                break

        if not placed:
            lines.append([block])

    for line in lines:
        line = sorted(line, key=lambda b: b.bbox[0])
        candidates.append((
            " ".join(b.text for b in line),
            (
                min(b.bbox[0] for b in line),
                min(b.bbox[1] for b in line),
                max(b.bbox[2] for b in line),
                max(b.bbox[3] for b in line),
            ),
        ))

    return candidates


def _find_best_match(
    aliases: Sequence[str],
    blocks: Sequence[OCRTextBlock],
) -> Tuple[float, Optional[str], Optional[BBox]]:
    candidates = _build_candidates(blocks)
    best_score = 0.0
    best_text = None
    best_bbox = None

    for alias in aliases:
        target = _normalize(alias)
        if not target:
            continue

        for candidate_text, bbox in candidates:
            candidate = _normalize(candidate_text)
            if not candidate:
                continue

            score = fuzz.token_set_ratio(target, candidate) / 100.0

            if target in candidate or candidate in target:
                score = max(score, 0.90)

            if score > best_score:
                best_score = score
                best_text = candidate_text
                best_bbox = bbox

    return best_score, best_text, best_bbox


def _region_bbox(region: Dict[str, Any], width: int, height: int) -> BBox:
    x = float(region.get("x", 0))
    y = float(region.get("y", 0))
    w = float(region.get("width", 1))
    h = float(region.get("height", 1))

    return (
        max(0, int(x * width)),
        max(0, int(y * height)),
        min(width, int((x + w) * width)),
        min(height, int((y + h) * height)),
    )


def _check_ink(
    gray: np.ndarray,
    bbox: BBox,
    minimum_ink: float,
) -> Tuple[bool, float, str]:
    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        return False, 0.0, "Invalid visual region."

    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return False, 0.0, "Empty visual region."

    _, binary = cv2.threshold(
        crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    ink = float(np.count_nonzero(binary)) / max(binary.size, 1) * 100.0

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    strokes = [c for c in contours if cv2.contourArea(c) > 3]

    passed = ink >= minimum_ink
    return passed, round(ink, 2), (
        f"Ink coverage: {ink:.2f}% ({len(strokes)} dark regions)."
    )


def _check_geometry(
    page_gray: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[bool, float, str]:
    geometry = config.get("page_geometry", {})
    expected = geometry.get("aspect_ratio")

    if not expected:
        return True, 1.0, "No saved page geometry."

    h, w = page_gray.shape[:2]
    actual = w / max(h, 1)
    expected = float(expected)
    tolerance = float(geometry.get("aspect_ratio_tolerance", 0.12))

    relative_difference = abs(actual - expected) / max(expected, 0.001)
    passed = relative_difference <= tolerance

    expected_orientation = geometry.get("orientation")
    actual_orientation = "landscape" if actual > 1.05 else "portrait"

    if expected_orientation and actual_orientation != expected_orientation:
        passed = False

    return passed, max(0.0, 1.0 - relative_difference), (
        f"Expected ratio {expected:.3f}; actual {actual:.3f}; "
        f"expected orientation {expected_orientation or 'any'}; "
        f"actual orientation {actual_orientation}."
    )


def validate_document(
    page_gray,
    blocks: Sequence[OCRTextBlock],
    config: Dict[str, Any],
) -> GenericValidation:

    checks: List[GenericCheck] = []
    missing_items: List[str] = []
    matched_anchors: List[str] = []
    missing_anchors: List[str] = []
    warnings: List[str] = []

    anchors = config.get("structure", {}).get("anchors", [])
    threshold = float(
        config.get("validation", {}).get("ocr_match_threshold", 0.70)
    )

    # 1. Orientation / page shape
    geometry_passed, geometry_confidence, geometry_details = _check_geometry(
        page_gray, config
    )

    checks.append(GenericCheck(
        id="page_geometry",
        label="Page orientation / format",
        check_type="geometry",
        required=True,
        passed=geometry_passed,
        confidence=round(geometry_confidence, 4),
        details=geometry_details,
    ))

    if not geometry_passed:
        missing_items.append("Page orientation / format")
        warnings.append(
            "Page orientation or aspect ratio does not match the selected template."
        )

    # 2. Learned headings / anchors
    for anchor in anchors:
        label = anchor.get("label", "Required field")
        aliases = anchor.get("aliases") or [label]

        score, matched_text, bbox = _find_best_match(aliases, blocks)
        passed = score >= threshold

        checks.append(GenericCheck(
            id=anchor.get("id", label),
            label=label,
            check_type="heading",
            required=True,
            passed=passed,
            confidence=round(score, 4),
            matched_text=matched_text if passed else None,
            bbox=bbox if passed else None,
            details=(
                "Template anchor detected."
                if passed
                else f"Template anchor not detected. Best similarity: {score:.2f}."
            ),
        ))

        if passed:
            matched_anchors.append(label)
        else:
            missing_anchors.append(label)
            missing_items.append(label)

    if anchors:
        structure_score = len(matched_anchors) / len(anchors) * 100.0
        classification_confidence = len(matched_anchors) / len(anchors)
    else:
        structure_score = 0.0
        classification_confidence = 0.0
        warnings.append("Selected template has no learned structural anchors.")

    # A rotated page must not pass merely because OCR can read it.
    if not geometry_passed:
        structure_score *= 0.50
        classification_confidence *= 0.50

    # 3. Signature / stamp
    visual_checks = [
        v for v in config.get("visual_checks", [])
        if v.get("required", True)
    ]

    height, width = page_gray.shape[:2]
    gray = page_gray if len(page_gray.shape) == 2 else cv2.cvtColor(
        page_gray, cv2.COLOR_BGR2GRAY
    )

    visual_passed = 0

    for visual in visual_checks:
        label = visual.get("label", visual.get("id", "Visual check"))
        bbox = _region_bbox(visual.get("region", {}), width, height)

        passed, ink, details = _check_ink(
            gray,
            bbox,
            float(visual.get("min_ink_percentage", 0.8)),
        )

        checks.append(GenericCheck(
            id=visual.get("id", label),
            label=label,
            check_type=visual.get("type", "visual"),
            required=True,
            passed=passed,
            confidence=min(1.0, ink / 5.0),
            bbox=bbox,
            details=details,
        ))

        if passed:
            visual_passed += 1
        else:
            missing_items.append(label)

    visual_score = (
        visual_passed / len(visual_checks) * 100.0
        if visual_checks else 100.0
    )

    # 4. Final score
    scoring = config.get("scoring", {})
    structure_weight = float(scoring.get("structure_weight", 80))
    visual_weight = float(scoring.get("visual_weight", 20))
    total = max(structure_weight + visual_weight, 1.0)

    overall_score = round(
        (
            structure_score * structure_weight
            + visual_score * visual_weight
        ) / total,
        2,
    )

    return GenericValidation(
        document_type=config.get("form_name", "Configured Form"),
        classification_confidence=round(classification_confidence, 4),
        checks=checks,
        structure_score=round(structure_score, 2),
        visual_score=round(visual_score, 2),
        overall_score=overall_score,
        missing_items=missing_items,
        warnings=warnings,
        matched_anchors=matched_anchors,
        missing_anchors=missing_anchors,
        geometry_passed=geometry_passed,
        geometry_details=geometry_details,
    )
