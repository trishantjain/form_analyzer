"""Config-driven CSR document classification and format validation.

This module intentionally separates *document structure* from the values
written by a person. A valid CSR can have different names, dates, numbers,
handwriting, signatures and stamps; the validator therefore checks labels,
sections, structural anchors and configured required visual regions instead of
matching the entire page pixel-for-pixel.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz

from modules.ocr_engine import BBox, OCRTextBlock
from modules.signature_detector import SignatureDetectionResult, analyze_signature_region


@dataclass
class CheckResult:
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
class DocumentClassification:
    document_type: str
    confidence: float
    passed: bool
    matched_markers: List[str] = field(default_factory=list)
    missing_markers: List[str] = field(default_factory=list)
    details: str = ""


@dataclass
class CSRValidation:
    classification: DocumentClassification
    checks: List[CheckResult]
    format_score: float
    required_checks_passed: int
    required_checks_total: int
    missing_items: List[str]
    warnings: List[str]

    @property
    def format_passed(self) -> bool:
        structural = [
            c for c in self.checks
            if c.required and c.check_type not in {"signature", "signature_and_stamp_region", "signature_and_seal_region", "stamp"}
        ]
        return all(c.passed for c in structural) if structural else True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": asdict(self.classification),
            "checks": [c.to_dict() for c in self.checks],
            "format_score": self.format_score,
            "required_checks_passed": self.required_checks_passed,
            "required_checks_total": self.required_checks_total,
            "missing_items": self.missing_items,
            "warnings": self.warnings,
        }


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _all_text(blocks: Sequence[OCRTextBlock]) -> str:
    return "\n".join(b.text for b in blocks)


def _candidate_lines(blocks: Sequence[OCRTextBlock]) -> List[Tuple[str, BBox]]:
    """Create OCR line candidates as well as individual OCR blocks."""
    candidates = [(b.text, b.bbox) for b in blocks if b.text.strip()]
    sorted_blocks = sorted([b for b in blocks if b.text.strip()], key=lambda b: (b.bbox[1], b.bbox[0]))
    lines: List[List[OCRTextBlock]] = []
    for block in sorted_blocks:
        cy = (block.bbox[1] + block.bbox[3]) / 2
        placed = False
        for line in lines:
            line_cy = sum((b.bbox[1] + b.bbox[3]) / 2 for b in line) / len(line)
            if abs(cy - line_cy) <= 18:
                line.append(block)
                placed = True
                break
        if not placed:
            lines.append([block])
    for line in lines:
        line = sorted(line, key=lambda b: b.bbox[0])
        text = " ".join(b.text for b in line)
        bbox = (min(b.bbox[0] for b in line), min(b.bbox[1] for b in line), max(b.bbox[2] for b in line), max(b.bbox[3] for b in line))
        candidates.append((text, bbox))
    return candidates


def _best_match(
    aliases: Sequence[str],
    blocks: Sequence[OCRTextBlock],
    threshold: float,
) -> Tuple[float, Optional[str], Optional[BBox]]:
    best_score = 0.0
    best_text = None
    best_bbox = None
    candidates = _candidate_lines(blocks)
    for alias in aliases:
        target = _normalize(alias)
        if not target:
            continue
        for text, bbox in candidates:
            candidate = _normalize(text)
            if not candidate:
                continue
            score = fuzz.token_set_ratio(target, candidate) / 100.0
            if target in candidate or candidate in target:
                score = max(score, 0.94 if target in candidate else 0.88)
            if score > best_score:
                best_score = score
                best_text = text
                best_bbox = bbox
    return best_score, best_text, best_bbox


def _match_groups(
    groups: Sequence[Dict[str, Any]],
    blocks: Sequence[OCRTextBlock],
    threshold: float,
) -> Tuple[List[str], List[str]]:
    matched: List[str] = []
    missing: List[str] = []
    for group in groups:
        label = group.get("label", "marker")
        aliases = group.get("aliases", [label])
        score, _text, _bbox = _best_match(aliases, blocks, threshold)
        if score >= threshold:
            matched.append(label)
        else:
            missing.append(label)
    return matched, missing


def classify_document(
    blocks: Sequence[OCRTextBlock],
    config: Dict[str, Any],
) -> DocumentClassification:
    cfg = config.get("document_classification", {})
    threshold = float(cfg.get("marker_match_threshold", 0.72))
    groups = cfg.get("required_markers", [])
    min_groups = int(cfg.get("minimum_markers", max(1, min(3, len(groups)))))
    min_confidence = float(cfg.get("minimum_confidence", 0.70))

    matched, missing = _match_groups(groups, blocks, threshold)
    confidence = len(matched) / max(len(groups), 1)
    passed = len(matched) >= min_groups and confidence >= min_confidence

    return DocumentClassification(
        document_type=config.get("form_name", "Configured Form") if passed else "Other Document",
        confidence=round(confidence, 4),
        passed=passed,
        matched_markers=matched,
        missing_markers=missing,
        details=f"Matched {len(matched)}/{len(groups)} configured document markers.",
    )


def _normalized_bbox(region: Dict[str, Any], width: int, height: int) -> BBox:
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


def _ink_region_check(
    page_gray,
    check: Dict[str, Any],
) -> CheckResult:
    height, width = page_gray.shape[:2]
    bbox = _normalized_bbox(check.get("region", {}), width, height)
    min_ink = float(check.get("min_ink_percentage", 1.5))
    result = analyze_signature_region(page_gray, bbox, min_ink_percentage=min_ink)
    required = bool(check.get("required", True))
    return CheckResult(
        id=check.get("id", "ink_check"),
        label=check.get("label", check.get("id", "Ink region")),
        check_type=check.get("type", "signature"),
        required=required,
        passed=result.present,
        confidence=min(1.0, result.ink_percentage / max(min_ink * 2.0, 0.01)),
        bbox=result.bbox,
        details=result.warning or f"Ink coverage: {result.ink_percentage:.2f}%",
    )


def validate_csr_page(
    page_gray,
    blocks: Sequence[OCRTextBlock],
    config: Dict[str, Any],
) -> CSRValidation:
    classification = classify_document(blocks, config)
    checks: List[CheckResult] = []
    missing: List[str] = []
    warnings: List[str] = []

    if not classification.passed:
        return CSRValidation(
            classification=classification,
            checks=[],
            format_score=0.0,
            required_checks_passed=0,
            required_checks_total=0,
            missing_items=[],
            warnings=["Document does not meet the configured CSR classification threshold."],
        )

    threshold = float(config.get("validation", {}).get("field_match_threshold", 0.72))
    fields = config.get("format_checks", {}).get("required_fields", [])
    sections = config.get("format_checks", {}).get("required_sections", [])

    for item in [*sections, *fields]:
        aliases = item.get("aliases", [item.get("label", "")])
        score, matched_text, bbox = _best_match(aliases, blocks, threshold)
        passed = score >= threshold
        check = CheckResult(
            id=item.get("id", item.get("label", "check")),
            label=item.get("label", item.get("id", "Required item")),
            check_type=item.get("type", "field"),
            required=bool(item.get("required", True)),
            passed=passed,
            confidence=round(score, 4),
            matched_text=matched_text if passed else None,
            bbox=bbox if passed else None,
            details="Matched" if passed else "Required label/section not detected by OCR",
        )
        checks.append(check)
        if check.required and not passed:
            missing.append(check.label)

    # Configurable visual checks. Regions are normalized to page width/height.
    for check in config.get("format_checks", {}).get("visual_checks", []):
        check_result = _ink_region_check(page_gray, check)
        checks.append(check_result)
        if check_result.required and not check_result.passed:
            missing.append(check_result.label)

    structural_checks = [
        c for c in checks
        if c.required and c.check_type not in {"signature", "signature_and_stamp_region", "signature_and_seal_region", "stamp"}
    ]
    total = len(structural_checks)
    passed_count = sum(1 for c in structural_checks if c.passed)
    score = round((passed_count / max(total, 1)) * 100.0, 2)

    format_cfg = config.get("format_checks", {})
    pass_threshold = float(format_cfg.get("pass_score_threshold", 90))
    if score < pass_threshold:
        warnings.append(f"Format score {score}% is below configured threshold {pass_threshold}%.")

    return CSRValidation(
        classification=classification,
        checks=checks,
        format_score=score,
        required_checks_passed=sum(1 for c in checks if c.required and c.passed),
        required_checks_total=sum(1 for c in checks if c.required),
        missing_items=missing,
        warnings=warnings,
    )
