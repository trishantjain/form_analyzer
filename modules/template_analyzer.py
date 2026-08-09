"""Learn a reusable document template profile from multiple known-good images."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import json
import re

import cv2
import numpy as np

from modules.ocr_engine import OCREngine, OCRTextBlock


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_probably_value(text: str) -> bool:
    """Reject likely handwritten/value-only OCR blocks from structural anchors."""
    t = _norm(text)
    if not t or len(t) < 2:
        return True
    # Pure numbers, dates, phone-like values, or very short codes are normally values.
    if re.fullmatch(r"[\d\s./:-]+", t):
        return True
    if re.fullmatch(r"[a-z]?\d+[a-z]?", t) and len(t) <= 12:
        return True
    return False


def _looks_like_structural_label(text: str) -> bool:
    t = _norm(text)
    if _is_probably_value(text):
        return False
    words = t.split()
    if len(words) == 1 and len(t) < 4:
        return False
    # Strong signals that this is a printed form label/heading.
    signals = (
        "name", "date", "code", "number", "no", "address", "signature",
        "sign", "stamp", "seal", "remarks", "remark", "centre", "center",
        "candidate", "candidate", "shift", "designation", "department",
        "district", "state", "city", "agency", "representative", "representative",
        "superintendent", "declaration", "statement", "total", "photo",
        "qualification", "dob", "gender", "category", "roll", "registration",
        "application", "exam", "examination", "certificate", "service",
    )
    return len(t) >= 4 and (len(words) >= 2 or any(s in t for s in signals))


def _cluster_labels(label_records: List[Dict[str, Any]], n_refs: int) -> List[Dict[str, Any]]:
    """
    Merge similar labels seen across references. A label becomes 'common'
    when it appears in enough reference documents.
    """
    from rapidfuzz import fuzz

    clusters: List[Dict[str, Any]] = []
    for rec in label_records:
        text = rec["text"]
        norm = rec["normalized"]
        placed = False
        for cluster in clusters:
            score = fuzz.token_set_ratio(norm, cluster["representative"]) / 100.0
            if score >= 0.78:
                cluster["items"].append(rec)
                # Prefer the longest/cleanest observed wording.
                if len(text) > len(cluster["label"]):
                    cluster["label"] = text
                    cluster["representative"] = norm
                placed = True
                break
        if not placed:
            clusters.append({
                "label": text,
                "representative": norm,
                "items": [rec],
            })

    result = []
    for cluster in clusters:
        refs = sorted({x["reference"] for x in cluster["items"]})
        if len(refs) < max(1, min(3, n_refs)):
            continue

        xs = [x["x"] for x in cluster["items"]]
        ys = [x["y"] for x in cluster["items"]]
        result.append({
            "id": re.sub(r"[^a-z0-9]+", "_", cluster["representative"]).strip("_")[:80] or "anchor",
            "label": cluster["label"],
            "aliases": sorted({x["text"] for x in cluster["items"]})[:8],
            "reference_count": len(refs),
            "x": round(float(np.median(xs)), 4),
            "y": round(float(np.median(ys)), 4),
            "position_tolerance": 0.10,
        })
    return result


def _visual_regions_from_labels(anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    regions = []
    for a in anchors:
        label = _norm(a["label"])
        kind = None
        if any(x in label for x in ("signature", "sign of", "signed by", "sign")):
            kind = "signature"
        elif any(x in label for x in ("stamp", "seal")):
            kind = "stamp"
        if not kind:
            continue

        # Region starts near/below the printed label. Size is deliberately
        # generous because the references can contain different handwriting.
        regions.append({
            "id": f'{a["id"]}_{kind}',
            "label": a["label"],
            "type": kind,
            "required": True,
            "region": {
                "x": max(0.0, a["x"] - 0.10),
                "y": min(0.92, a["y"] + 0.02),
                "width": 0.30,
                "height": 0.12,
            },
            "min_ink_percentage": 0.8 if kind == "signature" else 0.5,
        })
    return regions


def learn_template_profile(
    reference_paths: Sequence[Path],
    form_name: str,
    minimum_reference_count: int = 3,
) -> Dict[str, Any]:
    """OCR all reference images and build a generic reusable profile."""
    if len(reference_paths) < 1:
        raise ValueError("At least one reference image is required.")

    ocr = OCREngine.get_instance()
    records: List[Dict[str, Any]] = []
    page_sizes: List[Tuple[int, int]] = []
    all_text: List[str] = []

    for ref_index, path in enumerate(reference_paths, start=1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        page_sizes.append((w, h))

        blocks = ocr.run(image)
        all_text.append("\n".join(b.text for b in blocks))

        for block in blocks:
            if not _looks_like_structural_label(block.text):
                continue
            records.append({
                "reference": ref_index,
                "text": block.text.strip(),
                "normalized": _norm(block.text),
                "x": ((block.bbox[0] + block.bbox[2]) / 2) / max(w, 1),
                "y": ((block.bbox[1] + block.bbox[3]) / 2) / max(h, 1),
            })

    if not page_sizes:
        raise ValueError("None of the reference images could be read.")

    anchors = _cluster_labels(records, len(page_sizes))
    visual_checks = _visual_regions_from_labels(anchors)

    # Generic profile: no form-specific CSR classification.
    profile = {
        "form_name": form_name,
        "template_version": "2.0",
        "template_type": "generic",
        "reference_count": len(page_sizes),
        "structure": {
            "anchors": anchors,
            "minimum_anchor_match": 0.70,
            "required_anchor_ratio": 0.70,
        },
        "visual_checks": visual_checks,
        "matching": {
            "enabled": True,
            "layout_threshold": 0.30,
            "warning_only": True,
        },
        "validation": {
            "ocr_match_threshold": 0.70,
            "require_anchor_ratio": 0.70,
            "require_all_mandatory": False,
        },
        "scoring": {
            "structure_weight": 70,
            "visual_weight": 30,
            "pass_score_threshold": 75,
        },
        "metadata": {
            "learned_from": [p.name for p in reference_paths],
            "reference_text_sample": all_text[:3],
        },
    }
    return profile
