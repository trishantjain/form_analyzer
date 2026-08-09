"""Heading validation: fuzzy-match required headings against OCR text lines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from rapidfuzz import fuzz, process

from modules.ocr_engine import BBox, OCRTextBlock

STATUS_FOUND = "found"
STATUS_PROBABLY_FOUND = "probably_found"
STATUS_LOW_CONFIDENCE = "low_confidence"
STATUS_MISSING = "missing"


@dataclass
class HeadingResult:
    """Result of matching one required heading against OCR output."""

    heading: str
    matched_text: Optional[str]
    similarity: float
    confidence: float
    status: str
    page: Optional[int]
    bbox: Optional[BBox]


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise for matching."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_headings(
    required_headings: List[str],
    lines_by_page: List[List[str]],
    blocks_by_page: List[List[OCRTextBlock]],
    heading_match_threshold: float = 0.75,
    min_ocr_confidence: float = 0.60,
) -> List[HeadingResult]:
    """Fuzzy-match each required heading against OCR text lines across pages.

    Args:
        required_headings: Headings from the form template config.
        lines_by_page: Output of layout_analysis.group_text_lines(), one
            list of joined lines per page.
        blocks_by_page: Raw OCR blocks per page, used to locate a bbox and
            confidence for the best-matching line.
        heading_match_threshold: Minimum fuzzy similarity (0-1) to count a
            heading as "found" or "probably_found".
        min_ocr_confidence: Below this OCR confidence, a found heading is
            downgraded to "low_confidence".

    Returns:
        List of HeadingResult, one per required heading.
    """
    results: List[HeadingResult] = []

    # Build a flat searchable list of (normalized_line, page_index, original_line)
    all_lines = []
    for page_index, lines in enumerate(lines_by_page):
        for line in lines:
            norm = _normalize(line)
            if norm:
                all_lines.append((norm, page_index, line))

    choices = [entry[0] for entry in all_lines]

    for heading in required_headings:
        norm_heading = _normalize(heading)

        if not choices:
            results.append(
                HeadingResult(
                    heading=heading,
                    matched_text=None,
                    similarity=0.0,
                    confidence=0.0,
                    status=STATUS_MISSING,
                    page=None,
                    bbox=None,
                )
            )
            continue

        match = process.extractOne(
            norm_heading, choices, scorer=fuzz.token_sort_ratio
        )

        if match is None:
            results.append(
                HeadingResult(
                    heading=heading,
                    matched_text=None,
                    similarity=0.0,
                    confidence=0.0,
                    status=STATUS_MISSING,
                    page=None,
                    bbox=None,
                )
            )
            continue

        matched_norm, score, choice_index = match
        similarity = score / 100.0
        _norm, page_index, original_line = all_lines[choice_index]

        # Find the best bbox/confidence among blocks on that page whose text
        # contributes to the matched line.
        bbox, block_confidence = _find_bbox_and_confidence(
            original_line, blocks_by_page[page_index] if page_index < len(blocks_by_page) else []
        )

        status = STATUS_MISSING
        if similarity >= heading_match_threshold:
            if block_confidence is not None and block_confidence < min_ocr_confidence:
                status = STATUS_LOW_CONFIDENCE
            elif similarity >= 0.90:
                status = STATUS_FOUND
            else:
                status = STATUS_PROBABLY_FOUND

        results.append(
            HeadingResult(
                heading=heading,
                matched_text=original_line if similarity >= heading_match_threshold else None,
                similarity=round(similarity, 4),
                confidence=round(block_confidence, 4) if block_confidence is not None else 0.0,
                status=status,
                page=page_index + 1 if similarity >= heading_match_threshold else None,
                bbox=bbox if similarity >= heading_match_threshold else None,
            )
        )

    return results


def _find_bbox_and_confidence(line_text: str, page_blocks: List[OCRTextBlock]):
    """Find a bounding box covering the OCR blocks that make up a matched line
    and return their average confidence.
    """
    words = set(_normalize(line_text).split())
    matching_blocks = [
        b for b in page_blocks if _normalize(b.text) and _normalize(b.text) in words
        or any(w in _normalize(b.text) for w in words if len(w) > 2)
    ]

    if not matching_blocks:
        return None, None

    x_min = min(b.bbox[0] for b in matching_blocks)
    y_min = min(b.bbox[1] for b in matching_blocks)
    x_max = max(b.bbox[2] for b in matching_blocks)
    y_max = max(b.bbox[3] for b in matching_blocks)
    avg_confidence = sum(b.confidence for b in matching_blocks) / len(matching_blocks)

    return (x_min, y_min, x_max, y_max), avg_confidence
