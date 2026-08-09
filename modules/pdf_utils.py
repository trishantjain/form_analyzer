"""PDF handling utilities.

Converts PDF files into a list of PIL images (one per page) using PyMuPDF
(fitz). PyMuPDF is used instead of pdf2image/poppler because it has no
external system dependency (poppler-utils), which keeps installation fully
free and simple across Windows, Linux, and macOS.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Union

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

# Cap on the number of pages processed per document to avoid runaway memory
# use on malformed or huge files.
MAX_PAGES = 25

# Render resolution. 200 DPI is a good balance between OCR accuracy and
# processing speed for typical A4 / Letter forms.
RENDER_DPI = 200


class PDFProcessingError(Exception):
    """Raised when a PDF file cannot be read or converted."""


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> List[Image.Image]:
    """Convert raw PDF bytes into a list of PIL Images, one per page.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.
        dpi: Rendering resolution in dots per inch.

    Returns:
        List of PIL Image objects, one per page (RGB).

    Raises:
        PDFProcessingError: If the PDF is corrupt, empty, or cannot be opened.
    """
    if not pdf_bytes:
        raise PDFProcessingError("The uploaded PDF file is empty.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - we want to catch any fitz error
        raise PDFProcessingError(f"Could not open PDF file: {exc}") from exc

    if doc.page_count == 0:
        doc.close()
        raise PDFProcessingError("The PDF file contains no pages.")

    if doc.needs_pass:
        doc.close()
        raise PDFProcessingError("The PDF file is password-protected.")

    images: List[Image.Image] = []
    zoom = dpi / 72.0  # PDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    page_count = min(doc.page_count, MAX_PAGES)
    if doc.page_count > MAX_PAGES:
        logger.warning(
            "PDF has %s pages; only the first %s will be processed.",
            doc.page_count,
            MAX_PAGES,
        )

    try:
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            try:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
            except Exception as exc:  # noqa: BLE001
                raise PDFProcessingError(
                    f"Failed to render page {page_index + 1}: {exc}"
                ) from exc
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(image)
    finally:
        doc.close()

    if not images:
        raise PDFProcessingError("No pages could be rendered from the PDF.")

    return images


def load_file_as_images(filename: str, file_bytes: bytes) -> List[Image.Image]:
    """Load an uploaded file (PDF or image) into a list of PIL images.

    Args:
        filename: Original filename, used to determine file type by extension.
        file_bytes: Raw file bytes.

    Returns:
        List of PIL Images. Non-PDF images return a single-element list.

    Raises:
        PDFProcessingError: If a PDF cannot be processed.
        ValueError: If the file type is unsupported or the image is corrupt.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return pdf_bytes_to_images(file_bytes)

    if suffix in (".png", ".jpg", ".jpeg"):
        if not file_bytes:
            raise ValueError(f"The uploaded file '{filename}' is empty.")
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not read image file '{filename}': {exc}") from exc
        return [image]

    raise ValueError(
        f"Unsupported file type '{suffix}' for '{filename}'. "
        "Supported types: .pdf, .png, .jpg, .jpeg"
    )
