"""Generic validator for any saved form template."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

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

    matched_anchors: List[str]
    missing_anchors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "classification_confidence": self.classification_confidence,
            "checks": [
                check.to_dict()
                for check in self.checks
            ],
            "structure_score": self.structure_score,
            "visual_score": self.visual_score,
            "overall_score": self.overall_score,
            "missing_items": self.missing_items,
            "warnings": self.warnings,
            "matched_anchors": self.matched_anchors,
            "missing_anchors": self.missing_anchors,
        }


def _normalize(text: str) -> str:
    """Normalize OCR text for fuzzy comparison."""

    import re

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _build_ocr_candidates(
    blocks: Sequence[OCRTextBlock],
):
    """
    Build OCR candidates from individual OCR blocks
    and nearby OCR blocks combined into lines.
    """

    candidates = []

    valid_blocks = [
        block
        for block in blocks
        if block.text and block.text.strip()
    ]

    # Individual OCR blocks
    for block in valid_blocks:

        candidates.append(
            (
                block.text,
                block.bbox,
            )
        )

    # Sort blocks top-to-bottom and left-to-right
    sorted_blocks = sorted(
        valid_blocks,
        key=lambda block: (
            (block.bbox[1] + block.bbox[3]) / 2,
            block.bbox[0],
        ),
    )

    lines: List[List[OCRTextBlock]] = []

    for block in sorted_blocks:

        center_y = (
            block.bbox[1]
            + block.bbox[3]
        ) / 2

        matched_line = None

        for line in lines:

            line_center_y = sum(
                (
                    b.bbox[1]
                    + b.bbox[3]
                ) / 2
                for b in line
            ) / len(line)

            if abs(
                center_y - line_center_y
            ) <= 20:

                matched_line = line
                break

        if matched_line is None:

            matched_line = []

            lines.append(
                matched_line
            )

        matched_line.append(
            block
        )

    # Combine nearby OCR blocks into lines
    for line in lines:

        line.sort(
            key=lambda block:
            block.bbox[0]
        )

        text = " ".join(
            block.text.strip()
            for block in line
        )

        bbox = (
            min(
                block.bbox[0]
                for block in line
            ),
            min(
                block.bbox[1]
                for block in line
            ),
            max(
                block.bbox[2]
                for block in line
            ),
            max(
                block.bbox[3]
                for block in line
            ),
        )

        candidates.append(
            (
                text,
                bbox,
            )
        )

    return candidates


def _find_best_match(
    aliases: Sequence[str],
    blocks: Sequence[OCRTextBlock],
    threshold: float,
):
    """
    Find the best OCR match for a template anchor.
    """

    candidates = _build_ocr_candidates(
        blocks
    )

    best_score = 0.0
    best_text = None
    best_bbox = None

    for alias in aliases:

        target = _normalize(
            alias
        )

        if not target:
            continue

        for candidate_text, bbox in candidates:

            candidate = _normalize(
                candidate_text
            )

            if not candidate:
                continue

            score = (
                fuzz.token_set_ratio(
                    target,
                    candidate,
                )
                / 100.0
            )

            # Strong match when one contains the other.
            if (
                target in candidate
                or candidate in target
            ):
                score = max(
                    score,
                    0.90,
                )

            if score > best_score:

                best_score = score
                best_text = candidate_text
                best_bbox = bbox

    return (
        best_score,
        best_text,
        best_bbox,
    )


def _get_region_bbox(
    region: Dict[str, Any],
    width: int,
    height: int,
):
    """
    Convert normalized template coordinates
    into actual image coordinates.
    """

    x = float(
        region.get("x", 0)
    )

    y = float(
        region.get("y", 0)
    )

    region_width = float(
        region.get("width", 1)
    )

    region_height = float(
        region.get("height", 1)
    )

    x1 = max(
        0,
        int(x * width),
    )

    y1 = max(
        0,
        int(y * height),
    )

    x2 = min(
        width,
        int(
            (x + region_width)
            * width
        ),
    )

    y2 = min(
        height,
        int(
            (y + region_height)
            * height
        ),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def _check_visual_region(
    image,
    bbox,
    minimum_ink_percentage: float,
):
    """
    Basic ink detection for signature/stamp regions.

    This does not identify whose signature it is.
    It only checks whether meaningful ink exists
    inside the expected region.
    """

    import cv2
    import numpy as np

    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:

        return (
            False,
            0.0,
            bbox,
            "Invalid visual region",
        )

    region = image[
        y1:y2,
        x1:x2,
    ]

    if region.size == 0:

        return (
            False,
            0.0,
            bbox,
            "Empty visual region",
        )

    if len(region.shape) == 3:

        gray = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2GRAY,
        )

    else:

        gray = region

    # Threshold darker pixels.
    _, thresholded = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV,
    )

    ink_pixels = np.count_nonzero(
        thresholded
    )

    total_pixels = (
        thresholded.shape[0]
        * thresholded.shape[1]
    )

    ink_percentage = (
        ink_pixels
        / max(total_pixels, 1)
        * 100
    )

    passed = (
        ink_percentage
        >= minimum_ink_percentage
    )

    return (
        passed,
        ink_percentage,
        bbox,
        (
            f"Ink coverage: "
            f"{ink_percentage:.2f}%"
        ),
    )


def validate_document(
    page_gray,
    blocks: Sequence[OCRTextBlock],
    config: Dict[str, Any],
) -> GenericValidation:
    """
    Validate any document against a saved generic template.

    This function does NOT know whether the form is:
        MPSC
        CSR
        Constable
        Exam A
        Exam B
        etc.

    It only uses the selected template profile.
    """

    checks: List[
        GenericCheck
    ] = []

    missing_items: List[str] = []

    matched_anchors: List[str] = []

    missing_anchors: List[str] = []

    warnings: List[str] = []

    structure_config = config.get(
        "structure",
        {},
    )

    anchors = structure_config.get(
        "anchors",
        [],
    )

    validation_config = config.get(
        "validation",
        {},
    )

    match_threshold = float(
        validation_config.get(
            "ocr_match_threshold",
            0.70,
        )
    )

    required_ratio = float(
        validation_config.get(
            "require_anchor_ratio",
            0.70,
        )
    )

    # -------------------------------------------------
    # 1. CHECK TEMPLATE HEADINGS / STRUCTURAL ANCHORS
    # -------------------------------------------------

    for anchor in anchors:

        label = anchor.get(
            "label",
            "Required field",
        )

        aliases = anchor.get(
            "aliases",
            [label],
        )

        score, matched_text, bbox = (
            _find_best_match(
                aliases,
                blocks,
                match_threshold,
            )
        )

        passed = (
            score >= match_threshold
        )

        check = GenericCheck(
            id=anchor.get(
                "id",
                label,
            ),

            label=label,

            check_type="heading",

            required=True,

            passed=passed,

            confidence=round(
                score,
                4,
            ),

            matched_text=(
                matched_text
                if passed
                else None
            ),

            bbox=(
                bbox
                if passed
                else None
            ),

            details=(
                "Template anchor detected"
                if passed
                else
                "Template anchor not detected"
            ),
        )

        checks.append(
            check
        )

        if passed:

            matched_anchors.append(
                label
            )

        else:

            missing_anchors.append(
                label
            )

            missing_items.append(
                label
            )

    # Structure score
    if anchors:

        structure_score = (
            len(matched_anchors)
            / len(anchors)
            * 100
        )

    else:

        # No learned anchors means
        # we cannot claim the format passed.
        structure_score = 0.0

        warnings.append(
            "Selected template has no "
            "learned structural anchors."
        )

    # Classification confidence is simply
    # how much of the learned structure matched.
    if anchors:

        classification_confidence = (
            len(matched_anchors)
            / len(anchors)
        )

    else:

        classification_confidence = 0.0

    # -------------------------------------------------
    # 2. SIGNATURE / STAMP CHECKS
    # -------------------------------------------------

    visual_checks = config.get(
        "visual_checks",
        [],
    )

    visual_required_count = 0

    visual_passed_count = 0

    height, width = page_gray.shape[:2]

    for visual in visual_checks:

        if not visual.get(
            "required",
            True,
        ):
            continue

        visual_required_count += 1

        label = visual.get(
            "label",
            visual.get(
                "id",
                "Visual check",
            ),
        )

        region = visual.get(
            "region",
            {},
        )

        bbox = _get_region_bbox(
            region,
            width,
            height,
        )

        minimum_ink = float(
            visual.get(
                "min_ink_percentage",
                0.8,
            )
        )

        passed, ink_percentage, detected_bbox, details = (
            _check_visual_region(
                page_gray,
                bbox,
                minimum_ink,
            )
        )

        if passed:

            visual_passed_count += 1

        else:

            missing_items.append(
                label
            )

        confidence = min(
            1.0,
            ink_percentage
            / max(
                minimum_ink * 2,
                0.01,
            ),
        )

        checks.append(
            GenericCheck(
                id=visual.get(
                    "id",
                    label,
                ),

                label=label,

                check_type=visual.get(
                    "type",
                    "visual",
                ),

                required=True,

                passed=passed,

                confidence=round(
                    confidence,
                    4,
                ),

                bbox=detected_bbox,

                details=details,
            )
        )

    if visual_required_count:

        visual_score = (
            visual_passed_count
            / visual_required_count
            * 100
        )

    else:

        visual_score = 100.0

    # -------------------------------------------------
    # 3. WARN IF FORMAT MATCH IS TOO LOW
    # -------------------------------------------------

    if (
        anchors
        and classification_confidence
        < required_ratio
    ):

        warnings.append(
            f"Only "
            f"{len(matched_anchors)}/"
            f"{len(anchors)} "
            f"template anchors matched."
        )

    # -------------------------------------------------
    # 4. FINAL SCORE
    # -------------------------------------------------

    scoring = config.get(
        "scoring",
        {},
    )

    structure_weight = float(
        scoring.get(
            "structure_weight",
            70,
        )
    )

    visual_weight = float(
        scoring.get(
            "visual_weight",
            30,
        )
    )

    total_weight = max(
        structure_weight
        + visual_weight,
        1,
    )

    overall_score = (
        (
            structure_score
            * structure_weight
        )
        +
        (
            visual_score
            * visual_weight
        )
    ) / total_weight

    overall_score = round(
        overall_score,
        2,
    )

    return GenericValidation(
        document_type=config.get(
            "form_name",
            "Configured Form",
        ),

        classification_confidence=round(
            classification_confidence,
            4,
        ),

        checks=checks,

        structure_score=round(
            structure_score,
            2,
        ),

        visual_score=round(
            visual_score,
            2,
        ),

        overall_score=overall_score,

        missing_items=missing_items,

        warnings=warnings,

        matched_anchors=matched_anchors,

        missing_anchors=missing_anchors,
    )
