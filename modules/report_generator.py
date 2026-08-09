"""Generic results, scoring, annotation and report export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd


COLOR_GREEN = (0, 170, 0)
COLOR_RED = (0, 0, 220)
COLOR_ORANGE = (0, 140, 255)


@dataclass
class FormValidationResult:
    filename: str
    overall_status: str

    score: float
    max_score: float

    pages: int

    document_type: str
    document_confidence: float

    format_status: str
    format_score: float

    required_fields_passed: int
    required_fields_total: int

    signatures_status: str
    stamps_status: str

    missing_headings: List[str]
    missing_elements: List[str]

    low_confidence_items: List[str]

    detected_photo: bool
    detected_stamp: bool
    detected_signature: bool

    detected_checkboxes: int

    ocr_text: str

    warnings: List[str]

    checks: List[Dict[str, Any]] = field(
        default_factory=list
    )

    heading_details: List[Dict[str, Any]] = field(
        default_factory=list
    )

    checkbox_details: List[Dict[str, Any]] = field(
        default_factory=list
    )

    processing_duration_seconds: float = 0.0

    validation_details: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_result(
    filename: str,
    pages: int,
    validation,
    ocr_text: str,
    processing_duration_seconds: float,
    scoring_config: Optional[Dict[str, Any]] = None,
) -> FormValidationResult:
    """
    Convert GenericValidation into the result structure
    used by the existing UI/export code.
    """

    scoring_config = scoring_config or {}

    checks = list(
        getattr(
            validation,
            "checks",
            [],
        )
    )

    required_checks = [
        check
        for check in checks
        if getattr(
            check,
            "required",
            True,
        )
    ]

    structure_checks = [
        check
        for check in required_checks
        if getattr(
            check,
            "check_type",
            "",
        ) == "heading"
    ]

    visual_checks = [
        check
        for check in required_checks
        if getattr(
            check,
            "check_type",
            "",
        )
        in {
            "signature",
            "stamp",
            "signature_and_stamp_region",
            "signature_and_seal_region",
            "visual",
        }
    ]

    # -------------------------------------------------
    # STRUCTURE SCORE
    # -------------------------------------------------

    if structure_checks:

        structure_passed = sum(
            1
            for check in structure_checks
            if check.passed
        )

        structure_score = (
            structure_passed
            / len(structure_checks)
            * 100
        )

    else:

        structure_passed = 0
        structure_score = 0.0

    # -------------------------------------------------
    # VISUAL SCORE
    # -------------------------------------------------

    if visual_checks:

        visual_passed = sum(
            1
            for check in visual_checks
            if check.passed
        )

        visual_score = (
            visual_passed
            / len(visual_checks)
            * 100
        )

    else:

        visual_passed = 0
        visual_score = 100.0

    # -------------------------------------------------
    # WEIGHTS
    # -------------------------------------------------

    structure_weight = float(
        scoring_config.get(
            "format_weight",
            70,
        )
    )

    visual_weight = float(
        scoring_config.get(
            "visual_weight",
            30,
        )
    )

    total_weight = max(
        structure_weight
        + visual_weight,
        1.0,
    )

    score = (
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

    score = round(
        score,
        2,
    )

    # -------------------------------------------------
    # PASS THRESHOLD
    # -------------------------------------------------

    pass_threshold = float(
        scoring_config.get(
            "pass_score_threshold",
            75,
        )
    )

    missing_items = list(
        getattr(
            validation,
            "missing_items",
            [],
        )
    )

    classification_confidence = float(
        getattr(
            validation,
            "classification_confidence",
            0.0,
        )
    )

    required_anchor_ratio = float(
        scoring_config.get(
            "require_anchor_ratio",
            0.70,
        )
    )

    structure_passed_requirement = (
        classification_confidence
        >= required_anchor_ratio
    )

    passed = (
        score >= pass_threshold
        and structure_passed_requirement
        and not missing_items
    )

    # -------------------------------------------------
    # SIGNATURE
    # -------------------------------------------------

    signature_checks = [
        check
        for check in checks
        if getattr(
            check,
            "check_type",
            "",
        ) == "signature"
    ]

    if not signature_checks:

        signatures_status = "N/A"

    else:

        signatures_status = (
            "PASS"
            if all(
                check.passed
                for check in signature_checks
            )
            else "FAIL"
        )

    # -------------------------------------------------
    # STAMP
    # -------------------------------------------------

    stamp_checks = [
        check
        for check in checks
        if getattr(
            check,
            "check_type",
            "",
        ) == "stamp"
    ]

    if not stamp_checks:

        stamps_status = "N/A"

    else:

        stamps_status = (
            "PASS"
            if all(
                check.passed
                for check in stamp_checks
            )
            else "FAIL"
        )

    # -------------------------------------------------
    # LOW CONFIDENCE
    # -------------------------------------------------

    low_confidence_items = [
        check.label
        for check in checks
        if (
            0
            < getattr(
                check,
                "confidence",
                0,
            )
            < 0.75
        )
    ]

    # -------------------------------------------------
    # MISSING HEADINGS
    # -------------------------------------------------

    missing_headings = [
        check.label
        for check in structure_checks
        if not check.passed
    ]

    # -------------------------------------------------
    # DETECTED SIGNATURE / STAMP
    # -------------------------------------------------

    detected_signature = any(
        check.passed
        for check in signature_checks
    )

    detected_stamp = any(
        check.passed
        for check in stamp_checks
    )

    # -------------------------------------------------
    # WARNINGS
    # -------------------------------------------------

    warnings = list(
        getattr(
            validation,
            "warnings",
            [],
        )
    )

    return FormValidationResult(

        filename=filename,

        overall_status=(
            "PASS"
            if passed
            else "FAIL"
        ),

        score=score,

        max_score=100.0,

        pages=pages,

        document_type=getattr(
            validation,
            "document_type",
            "Configured Form",
        ),

        document_confidence=round(
            classification_confidence
            * 100,
            2,
        ),

        format_status=(
            "PASS"
            if structure_passed_requirement
            else "FAIL"
        ),

        format_score=round(
            structure_score,
            2,
        ),

        required_fields_passed=(
            structure_passed
        ),

        required_fields_total=(
            len(structure_checks)
        ),

        signatures_status=(
            signatures_status
        ),

        stamps_status=(
            stamps_status
        ),

        missing_headings=(
            missing_headings
        ),

        missing_elements=(
            missing_items
        ),

        low_confidence_items=(
            low_confidence_items
        ),

        detected_photo=False,

        detected_stamp=(
            detected_stamp
        ),

        detected_signature=(
            detected_signature
        ),

        detected_checkboxes=0,

        ocr_text=ocr_text,

        warnings=warnings,

        checks=[
            check.to_dict()
            for check in checks
        ],

        heading_details=[
            check.to_dict()
            for check in structure_checks
        ],

        checkbox_details=[],

        processing_duration_seconds=(
            processing_duration_seconds
        ),

        validation_details=(
            validation.to_dict()
        ),
    )


def draw_annotations(
    image,
    validation,
    page_number: int = 1,
):
    """
    Draw generic validation results on an image.
    """

    annotated = image.copy()

    checks = getattr(
        validation,
        "checks",
        [],
    )

    for check in checks:

        bbox = getattr(
            check,
            "bbox",
            None,
        )

        if not bbox:
            continue

        if check.passed:

            color = COLOR_GREEN

        elif getattr(
            check,
            "confidence",
            0.0,
        ) > 0.4:

            color = COLOR_ORANGE

        else:

            color = COLOR_RED

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            annotated,
            str(
                check.label
            )[:32],
            (
                x1,
                max(
                    18,
                    y1 - 6,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def results_to_dataframe(
    results: List[FormValidationResult],
) -> pd.DataFrame:
    """
    Convert validation results into a table.
    """

    rows = []

    for result in results:

        rows.append(
            {
                "File": result.filename,

                "Document Type":
                    result.document_type,

                "Doc Confidence":
                    f"{result.document_confidence:.1f}%",

                "Format":
                    result.format_status,

                "Format Score":
                    f"{result.format_score:.1f}%",

                "Fields":
                    (
                        f"{result.required_fields_passed}/"
                        f"{result.required_fields_total}"
                    ),

                "Signature":
                    result.signatures_status,

                "Stamp / Seal":
                    result.stamps_status,

                "Final":
                    result.overall_status,

                "Score":
                    f"{result.score:.1f}%",
            }
        )

    return pd.DataFrame(
        rows
    )


def export_csv(
    results: List[FormValidationResult],
    output_path: Path,
) -> Path:
    """
    Export results to CSV.
    """

    dataframe = results_to_dataframe(
        results
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


def export_json(
    results: List[FormValidationResult],
    output_path: Path,
) -> Path:
    """
    Export detailed results to JSON.
    """

    output_path.write_text(
        json.dumps(
            [
                result.to_dict()
                for result in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path
