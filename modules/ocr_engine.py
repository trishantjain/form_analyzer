"""PaddleOCR wrapper.

PaddleOCR's Python API has changed across releases:
  * <= 2.6:  PaddleOCR(use_angle_cls=True, lang="en").ocr(img, cls=True)
             -> [[ [box, (text, confidence)], ... ]]   (one list per image)
  * 2.7.x:   Same call signature, but `use_angle_cls` began emitting a
             deprecation warning in favor of `use_textline_orientation`.
  * 3.x:     Introduces `.predict()` with a dict-based result schema
             (result[i]['rec_texts'], result[i]['rec_scores'],
             result[i]['rec_polys'], etc.) and further renames.

Rather than assume one exact schema, this module normalizes whatever the
installed version returns into a single OCRResult data structure. If the
installed PaddleOCR version returns something unrecognized, an explicit
OCRProcessingError is raised instead of crashing deep inside detection code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max


class OCRProcessingError(Exception):
    """Raised when OCR fails or the model output cannot be interpreted."""


@dataclass
class OCRTextBlock:
    """A single recognized text block."""

    text: str
    confidence: float
    bbox: BBox  # axis-aligned bounding box in pixel coordinates
    polygon: List[Tuple[int, int]]  # original (possibly rotated) polygon


class OCREngine:
    """Lazily-initialized, cached PaddleOCR wrapper."""

    _instance: Optional["OCREngine"] = None

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._ocr = None  # Initialized on first use (models are large).

    @classmethod
    def get_instance(cls, lang: str = "en") -> "OCREngine":
        """Return a process-wide singleton so models load only once."""
        if cls._instance is None:
            cls._instance = cls(lang=lang)
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR  # imported lazily: slow import
        except ImportError as exc:
            raise OCRProcessingError(
                "PaddleOCR is not installed. Run: pip install paddleocr paddlepaddle"
            ) from exc

        last_error: Optional[Exception] = None
        # Try the newer keyword first, then fall back for older versions.
        init_kwargs_attempts = [
            {"use_textline_orientation": True, "lang": self.lang},
            {"use_angle_cls": True, "lang": self.lang},
            {"lang": self.lang},
        ]
        for kwargs in init_kwargs_attempts:
            try:
                self._ocr = PaddleOCR(**kwargs)
                logger.info("PaddleOCR initialized with kwargs=%s", kwargs)
                return
            except TypeError as exc:
                last_error = exc
                continue
        raise OCRProcessingError(
            f"Failed to initialize PaddleOCR with any known API signature: {last_error}"
        )

    def run(self, image: np.ndarray) -> List[OCRTextBlock]:
        """Run OCR on an OpenCV BGR image and return normalized text blocks.

        Args:
            image: OpenCV BGR numpy array.

        Returns:
            List of OCRTextBlock, empty list if no text was detected.

        Raises:
            OCRProcessingError: If PaddleOCR is missing or its output cannot
                be parsed under any known schema.
        """
        self._ensure_loaded()

        try:
            raw_result = self._run_raw(image)
        except OCRProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise OCRProcessingError(f"OCR inference failed: {exc}") from exc

        return self._normalize_result(raw_result)

    def _run_raw(self, image: np.ndarray):
        """Call whichever inference method the installed PaddleOCR exposes."""
        if hasattr(self._ocr, "predict"):
            # PaddleOCR 3.x style
            try:
                return ("predict", self._ocr.predict(image))
            except Exception as exc:  # noqa: BLE001
                logger.warning(".predict() failed (%s); trying legacy .ocr()", exc)

        if hasattr(self._ocr, "ocr"):
            try:
                return ("ocr_cls", self._ocr.ocr(image, cls=True))
            except TypeError:
                return ("ocr_plain", self._ocr.ocr(image))

        raise OCRProcessingError(
            "Installed PaddleOCR version exposes neither .predict() nor .ocr(); "
            "please install a supported version, e.g. pip install paddleocr==2.7.3"
        )

    def _normalize_result(self, raw) -> List[OCRTextBlock]:
        """Normalize outputs from either the legacy .ocr() or new .predict() API."""
        mode, result = raw
        blocks: List[OCRTextBlock] = []

        if result is None:
            return blocks

        if mode.startswith("ocr"):
            # Legacy schema: result is a list (one entry per input image);
            # each entry is a list of [polygon, (text, confidence)].
            page_results = result[0] if result and isinstance(result[0], list) else result
            if not page_results:
                return blocks
            for item in page_results:
                try:
                    polygon_raw, (text, confidence) = item
                except (ValueError, TypeError):
                    continue
                polygon = [(int(p[0]), int(p[1])) for p in polygon_raw]
                bbox = _polygon_to_bbox(polygon)
                blocks.append(
                    OCRTextBlock(
                        text=str(text),
                        confidence=float(confidence),
                        bbox=bbox,
                        polygon=polygon,
                    )
                )
            return blocks

        if mode == "predict":
            # New schema: result is a list of dict-like page results.
            for page in result:
                page_dict = page if isinstance(page, dict) else getattr(page, "__dict__", {})
                texts = page_dict.get("rec_texts", [])
                scores = page_dict.get("rec_scores", [])
                polys = page_dict.get("rec_polys", page_dict.get("rec_boxes", []))
                for text, score, poly in zip(texts, scores, polys):
                    poly_arr = np.array(poly).reshape(-1, 2)
                    polygon = [(int(p[0]), int(p[1])) for p in poly_arr]
                    bbox = _polygon_to_bbox(polygon)
                    blocks.append(
                        OCRTextBlock(
                            text=str(text),
                            confidence=float(score),
                            bbox=bbox,
                            polygon=polygon,
                        )
                    )
            return blocks

        raise OCRProcessingError(f"Unrecognized OCR result schema (mode={mode}).")


def _polygon_to_bbox(polygon: List[Tuple[int, int]]) -> BBox:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def get_full_text(blocks: List[OCRTextBlock]) -> str:
    """Join all recognized text blocks into a single readable string."""
    return "\n".join(block.text for block in blocks)
