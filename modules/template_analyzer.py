"""Learn a reusable generic document template from known-good reference images."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import re

import cv2
import numpy as np
from rapidfuzz import fuzz

from modules.ocr_engine import OCREngine


def _norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_value(text: str) -> bool:
    t = _norm(text)
    if not t or len(t) < 2:
        return True
    if re.fullmatch(r"[\d\s./:-]+", t):
        return True
    if re.fullmatch(r"[a-z]?\d+[a-z]?", t) and len(t) <= 12:
        return True
    return False


def _looks_like_label(text: str) -> bool:
    t = _norm(text)
    if _is_value(text) or len(t) < 4:
        return False

    words = t.split()
    signals = (
        "name", "date", "code", "number", "no", "address", "signature",
        "sign", "stamp", "seal", "remarks", "remark", "centre", "center",
        "candidate", "shift", "designation", "department", "district",
        "state", "city", "agency", "representative", "superintendent",
        "declaration", "statement", "total", "photo", "qualification",
        "dob", "gender", "category", "roll", "registration", "application",
        "exam", "examination", "certificate", "service", "form", "father",
        "mother", "school", "college", "subject", "course", "university",
    )
    return len(words) >= 2 or any(s in t for s in signals)


def _cluster_labels(records: List[Dict[str, Any]], n_refs: int,
                    minimum_reference_count: int) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []

    for rec in records:
        placed = False
        for cluster in clusters:
            score = fuzz.token_set_ratio(
                rec["normalized"], cluster["representative"]
            ) / 100.0
            if score >= 0.78:
                cluster["items"].append(rec)
                if len(rec["text"]) > len(cluster["label"]):
                    cluster["label"] = rec["text"]
                    cluster["representative"] = rec["normalized"]
                placed = True
                break

        if not placed:
            clusters.append({
                "label": rec["text"],
                "representative": rec["normalized"],
                "items": [rec],
            })

    required_refs = min(max(1, minimum_reference_count), max(1, n_refs))
    result = []

    for cluster in clusters:
        refs = sorted({x["reference"] for x in cluster["items"]})
        if len(refs) < required_refs:
            continue

        xs = [x["x"] for x in cluster["items"]]
        ys = [x["y"] for x in cluster["items"]]

        result.append({
            "id": re.sub(
                r"[^a-z0-9]+", "_", cluster["representative"]
            ).strip("_")[:80] or "anchor",
            "label": cluster["label"],
            "aliases": sorted({x["text"] for x in cluster["items"]})[:10],
            "reference_count": len(refs),
            "x": round(float(np.median(xs)), 4),
            "y": round(float(np.median(ys)), 4),
            "position_tolerance": 0.12,
        })

    return result


def _visual_regions_from_labels(anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    regions = []

    for a in anchors:
        label = _norm(a["label"])

        if any(x in label for x in ("signature", "sign of", "signed by", "sign")):
            kind = "signature"
        elif any(x in label for x in ("stamp", "seal")):
            kind = "stamp"
        else:
            continue

        regions.append({
            "id": f'{a["id"]}_{kind}',
            "label": a["label"],
            "type": kind,
            "required": True,
            "region": {
                "x": max(0.0, a["x"] - 0.10),
                "y": min(0.88, a["y"] + 0.02),
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
            text = block.text.strip()
            if not _looks_like_label(text):
                continue

            records.append({
                "reference": ref_index,
                "text": text,
                "normalized": _norm(text),
                "x": ((block.bbox[0] + block.bbox[2]) / 2) / max(w, 1),
                "y": ((block.bbox[1] + block.bbox[3]) / 2) / max(h, 1),
            })

    if not page_sizes:
        raise ValueError("None of the reference images could be read.")

    anchors = _cluster_labels(
        records, len(page_sizes), minimum_reference_count
    )

    ratios = [w / max(h, 1) for w, h in page_sizes]
    median_ratio = float(np.median(ratios))
    orientation = "landscape" if median_ratio > 1.05 else "portrait"

    return {
        "form_name": form_name,
        "template_version": "3.0",
        "template_type": "generic",
        "reference_count": len(page_sizes),

        # This is what prevents a 90-degree rotated document from passing.
        "page_geometry": {
            "aspect_ratio": round(median_ratio, 5),
            "orientation": orientation,
            "aspect_ratio_tolerance": 0.12,
            "reference_sizes": [
                {"width": w, "height": h} for w, h in page_sizes
            ],
        },

        "structure": {
            "anchors": anchors,
            "minimum_anchor_match": 0.70,
            "required_anchor_ratio": 0.70,
        },

        "visual_checks": _visual_regions_from_labels(anchors),

        "matching": {
            "enabled": True,
            "layout_threshold": 0.30,
            "warning_only": False,
        },

        "validation": {
            "ocr_match_threshold": 0.70,
            "require_anchor_ratio": 0.70,
            "require_all_mandatory": True,
        },

        "scoring": {
            "structure_weight": 80,
            "visual_weight": 20,
            "pass_score_threshold": 75,
        },

        "metadata": {
            "learned_from": [p.name for p in reference_paths],
            "reference_text_sample": all_text[:3],
        },
    }
