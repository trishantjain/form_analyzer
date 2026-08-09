"""Results, scoring, annotation and report export for the CSR validator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd

from modules.csr_validator import CSRValidation
from modules.ocr_engine import BBox

COLOR_GREEN = (0, 170, 0)
COLOR_RED = (0, 0, 220)
COLOR_ORANGE = (0, 140, 255)
COLOR_BLUE = (220, 130, 0)


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
    checks: List[Dict[str, Any]] = field(default_factory=list)
    heading_details: List[Dict[str, Any]] = field(default_factory=list)
    checkbox_details: List[Dict[str, Any]] = field(default_factory=list)
    processing_duration_seconds: float = 0.0
    validation_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _visual_status(validation: CSRValidation, types: set[str]) -> tuple[str, int, int]:
    relevant = [c for c in validation.checks if c.check_type in types and c.required]
    if not relevant:
        return "N/A", 0, 0
    passed = sum(1 for c in relevant if c.passed)
    return ("PASS" if passed == len(relevant) else "FAIL", passed, len(relevant))


def build_result(
    filename: str,
    pages: int,
    validation: CSRValidation,
    ocr_text: str,
    processing_duration_seconds: float,
    scoring_config: Dict[str, Any],
) -> FormValidationResult:
    structural = [
        c for c in validation.checks
        if c.required and c.check_type not in {"signature", "signature_and_stamp_region", "signature_and_seal_region", "stamp"}
    ]
    visual = [
        c for c in validation.checks
        if c.required and c.check_type in {"signature", "signature_and_stamp_region", "signature_and_seal_region", "stamp"}
    ]

    structure_score = (sum(c.passed for c in structural) / max(len(structural), 1)) * 100
    visual_score = (sum(c.passed for c in visual) / max(len(visual), 1)) * 100 if visual else 100.0
    classification_score = validation.classification.confidence * 100

    cfg = {
        "classification_weight": 20,
        "format_weight": 60,
        "visual_weight": 20,
        "pass_score_threshold": 85,
        "require_all_mandatory_for_pass": True,
        **(scoring_config or {}),
    }
    total_weight = max(cfg["classification_weight"] + cfg["format_weight"] + cfg["visual_weight"], 1)
    score = (
        classification_score * cfg["classification_weight"]
        + structure_score * cfg["format_weight"]
        + visual_score * cfg["visual_weight"]
    ) / total_weight
    score = round(score, 2)

    missing = list(validation.missing_items)
    mandatory_ok = not missing and validation.classification.passed
    passed = score >= cfg["pass_score_threshold"] and (
        mandatory_ok if cfg["require_all_mandatory_for_pass"] else True
    )

    sig_status, sig_passed, sig_total = _visual_status(
        validation, {"signature", "signature_and_stamp_region", "signature_and_seal_region"}
    )
    stamp_checks = [
        c for c in validation.checks
        if c.required and c.check_type in {"signature_and_stamp_region", "stamp", "signature_and_seal_region"}
    ]
    stamps_status = "N/A" if not stamp_checks else ("PASS" if all(c.passed for c in stamp_checks) else "FAIL")

    return FormValidationResult(
        filename=filename,
        overall_status="PASS" if passed else "FAIL",
        score=score,
        max_score=100.0,
        pages=pages,
        document_type=validation.classification.document_type,
        document_confidence=round(validation.classification.confidence * 100, 2),
        format_status="PASS" if validation.format_passed else "FAIL",
        format_score=round(structure_score, 2),
        required_fields_passed=sum(1 for c in structural if c.passed),
        required_fields_total=len(structural),
        signatures_status=sig_status,
        stamps_status=stamps_status,
        missing_headings=[],
        missing_elements=missing,
        low_confidence_items=[c.label for c in validation.checks if c.required and 0 < c.confidence < 0.75],
        detected_photo=False,
        detected_stamp=any(c.passed and c.check_type in {"signature_and_stamp_region", "stamp"} for c in validation.checks),
        detected_signature=any(c.passed and c.check_type in {"signature", "signature_and_stamp_region", "signature_and_seal_region"} for c in validation.checks),
        detected_checkboxes=0,
        ocr_text=ocr_text,
        warnings=validation.warnings,
        checks=[c.to_dict() for c in validation.checks],
        validation_details=validation.to_dict() | {
            "classification_score": round(classification_score, 2),
            "structure_score": round(structure_score, 2),
            "visual_score": round(visual_score, 2),
            "weights": cfg,
        },
        processing_duration_seconds=processing_duration_seconds,
    )


def draw_annotations(
    image,
    validation: CSRValidation,
    page_number: int = 1,
):
    """Annotate configured OCR matches and visual regions."""
    annotated = image.copy()

    def box(bbox: Optional[BBox], color, label: str):
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, label[:32], (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

    for check in validation.checks:
        color = COLOR_GREEN if check.passed else (COLOR_ORANGE if check.confidence > 0.4 else COLOR_RED)
        box(check.bbox, color, check.label)

    return annotated


def results_to_dataframe(results: List[FormValidationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "File": r.filename,
            "Document Type": r.document_type,
            "Doc Confidence": f"{r.document_confidence:.1f}%",
            "Format": r.format_status,
            "Format Score": f"{r.format_score:.1f}%",
            "Fields": f"{r.required_fields_passed}/{r.required_fields_total}",
            "Signature": r.signatures_status,
            "Stamp / Seal": r.stamps_status,
            "Final": r.overall_status,
            "Score": f"{r.score:.1f}%",
        })
    return pd.DataFrame(rows)


def export_csv(results: List[FormValidationResult], output_path: Path) -> Path:
    df = results_to_dataframe(results)
    df.to_csv(output_path, index=False)
    return output_path


def export_json(results: List[FormValidationResult], output_path: Path) -> Path:
    output_path.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")
    return output_path
