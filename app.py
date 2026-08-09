"""Form Analyzer - Streamlit application entry point.

A fully local, free/open-source tool for checking whether scanned or
photographed student forms follow a predefined format (headings, photo,
stamp, signature, checkboxes, sections).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from modules.checkbox_detector import detect_checkboxes
from modules.database import FormDatabase
from modules.heading_validator import validate_headings
from modules.image_preprocessing import cv2_to_pil, pil_to_cv2, preprocess_page
from modules.layout_analysis import find_non_text_regions, group_text_lines
from modules.ocr_engine import OCREngine, OCRProcessingError, get_full_text
from modules.pdf_utils import PDFProcessingError, load_file_as_images
from modules.photo_detector import detect_photo_region
from modules.report_generator import (
    FormValidationResult,
    compute_score,
    draw_annotations,
    export_csv,
    export_json,
)
from modules.signature_detector import analyze_signature_region
from modules.stamp_detector import detect_stamp
from modules.template_manager import TemplateManager, DEFAULT_TEMPLATE

# --------------------------------------------------------------------------
# Logging: technical errors go to a log file, NOT the Streamlit UI.
# --------------------------------------------------------------------------
LOG_DIR = Path("output")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "form_analyzer.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("form_analyzer.app")

DEFAULT_CONFIG_PATH = Path("config/form_template.json")
OUTPUT_DIR = Path("output")
DB_PATH = OUTPUT_DIR / "form_analyzer.db"
TEMPLATE_ROOT = Path("templates/profiles")

st.set_page_config(page_title="Form Analyzer", layout="wide")


# --------------------------------------------------------------------------
# Template/profile helpers
# --------------------------------------------------------------------------
def init_session_state() -> None:
    if "template_manager" not in st.session_state:
        st.session_state.template_manager = TemplateManager(TEMPLATE_ROOT, DEFAULT_CONFIG_PATH)
    if "selected_template" not in st.session_state:
        templates = st.session_state.template_manager.list_templates()
        st.session_state.selected_template = templates[0]["slug"] if templates else None
    if "results" not in st.session_state:
        st.session_state.results: List[FormValidationResult] = []
    if "annotated_images" not in st.session_state:
        st.session_state.annotated_images: Dict[str, List[Image.Image]] = {}
    if "db" not in st.session_state:
        st.session_state.db = FormDatabase(DB_PATH)


def get_selected_template() -> Optional[Dict[str, Any]]:
    manager: TemplateManager = st.session_state.template_manager
    slug = st.session_state.get("selected_template")
    if not slug:
        return None
    try:
        config = manager.load(slug)
    except Exception:
        return None
    info = next((x for x in manager.list_templates() if x["slug"] == slug), None)
    return {"slug": slug, "config": config, "info": info}


def load_reference_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def get_template_references(slug: str) -> List[Path]:
    return st.session_state.template_manager.reference_files(slug)


def render_template_selector() -> Optional[Dict[str, Any]]:
    manager: TemplateManager = st.session_state.template_manager
    templates = manager.list_templates()
    if not templates:
        st.warning("No saved formats yet. Create one in Template Manager.")
        return None

    labels = [x["name"] for x in templates]
    current_slug = st.session_state.get("selected_template")
    current_index = next((i for i, x in enumerate(templates) if x["slug"] == current_slug), 0)
    selected_name = st.selectbox("Format to check", labels, index=current_index)
    selected = templates[labels.index(selected_name)]
    st.session_state.selected_template = selected["slug"]
    return {"slug": selected["slug"], "config": manager.load(selected["slug"]), "info": selected}


def render_template_manager() -> None:
    st.header("Template Manager")
    st.caption("Create and save reusable document formats. Each format has its own rules and reference images.")
    manager: TemplateManager = st.session_state.template_manager
    templates = manager.list_templates()

    if templates:
        st.subheader("Saved formats")
        rows = [{"Format": x["name"], "Reference images": x["reference_count"], "Folder": x["slug"]} for x in templates]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("Create a new format")
    with st.form("create_template_form"):
        new_name = st.text_input("Format name", placeholder="e.g. Exam A, Exam B, CSR 2027")
        base_options = ["Blank configuration"] + [x["name"] for x in templates]
        base_choice = st.selectbox("Start configuration from", base_options)
        reference_files = st.file_uploader(
            "Upload known-good template/reference images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="new_template_refs",
        )
        create_clicked = st.form_submit_button("Save New Format", type="primary")

    if create_clicked:
        if not new_name.strip():
            st.error("Enter a format name.")
        elif manager.exists(new_name):
            st.error("A format with this name already exists.")
        else:
            try:
                if base_choice == "Blank configuration":
                    base_config = DEFAULT_TEMPLATE
                else:
                    base_slug = next(x["slug"] for x in templates if x["name"] == base_choice)
                    base_config = manager.load(base_slug)
                slug = manager.create(new_name.strip(), base_config)
                if reference_files:
                    manager.add_references(slug, reference_files)
                st.session_state.selected_template = slug
                st.success(f"Format '{new_name.strip()}' saved successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create format: {exc}")

    templates = manager.list_templates()
    if not templates:
        return

    st.subheader("Manage an existing format")
    selected_name = st.selectbox("Format", [x["name"] for x in templates], key="manager_selected")
    selected = next(x for x in templates if x["name"] == selected_name)
    slug = selected["slug"]
    config = manager.load(slug)

    col1, col2, col3 = st.columns(3)
    with col1:
        new_name = st.text_input("Rename format", value=config.get("form_name", selected_name), key="rename_name")
        if st.button("Rename", key="rename_btn"):
            try:
                new_slug = manager.rename(slug, new_name.strip())
                st.session_state.selected_template = new_slug
                st.success("Format renamed.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with col2:
        more_refs = st.file_uploader("Add reference images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="more_refs")
        if st.button("Add References", key="add_refs_btn"):
            if not more_refs:
                st.warning("Select at least one image.")
            else:
                count = manager.add_references(slug, more_refs)
                st.success(f"Added {count} reference image(s).")
                st.rerun()
    with col3:
        st.write("Danger zone")
        if st.button("Delete Format", key="delete_template_btn", type="secondary"):
            if len(templates) == 1:
                st.error("Keep at least one format.")
            else:
                manager.delete(slug)
                remaining = manager.list_templates()
                st.session_state.selected_template = remaining[0]["slug"] if remaining else None
                st.success("Format deleted.")
                st.rerun()

    refs = manager.reference_files(slug)
    st.write(f"**{len(refs)} reference image(s)**")
    if refs:
        cols = st.columns(min(5, len(refs)))
        for i, ref in enumerate(refs):
            with cols[i % len(cols)]:
                img = load_reference_image(ref)
                if img is not None:
                    st.image(img, caption=ref.name, width=150)


def render_template_configuration() -> None:
    st.header("Template Configuration")
    st.caption("Rules are stored per format. You can change them without editing Python code.")
    selected = render_template_selector()
    if not selected:
        return
    manager: TemplateManager = st.session_state.template_manager
    slug = selected["slug"]
    config = selected["config"]

    st.info(f"Editing: **{config.get('form_name', slug)}**")
    edited = st.text_area("Template JSON", value=json.dumps(config, indent=2, ensure_ascii=False), height=600)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Configuration", type="primary"):
            try:
                parsed = json.loads(edited)
                if not isinstance(parsed, dict):
                    raise ValueError("Configuration root must be a JSON object.")
                parsed["form_name"] = config.get("form_name", slug)
                manager.save(slug, parsed)
                st.success("Configuration saved.")
                st.rerun()
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
            except Exception as exc:
                st.error(str(exc))
    with col2:
        if st.button("Reset to Default", type="secondary"):
            manager.save(slug, {**DEFAULT_TEMPLATE, "form_name": config.get("form_name", slug)})
            st.success("Configuration reset.")
            st.rerun()

# --------------------------------------------------------------------------
# Core per-file analysis pipeline
# --------------------------------------------------------------------------
def analyze_single_file(
    filename: str,
    file_bytes: bytes,
    config: Dict[str, Any],
    stamp_reference: Optional[Image.Image],
    logo_reference: Optional[Image.Image],
) -> tuple[Optional[FormValidationResult], List[Image.Image]]:
    """Run the full detection pipeline on one uploaded file.

    Returns (result, annotated_page_images). result is None if the file
    could not be processed at all (error already shown to user).
    """
    start_time = time.time()
    annotated_pages: List[Image.Image] = []

    try:
        pages = load_file_as_images(filename, file_bytes)
    except (PDFProcessingError, ValueError) as exc:
        st.error(f"'{filename}': {exc}")
        logger.warning("File load failed for %s: %s", filename, exc)
        return None, []
    except Exception as exc:  # noqa: BLE001
        st.error(f"'{filename}': an unexpected error occurred while loading the file.")
        logger.error("Unexpected load error for %s: %s\n%s", filename, exc, traceback.format_exc())
        return None, []

    ocr_engine = OCREngine.get_instance()

    all_heading_results = []
    all_lines_by_page: List[List[str]] = []
    all_blocks_by_page = []
    combined_ocr_text_parts: List[str] = []

    photo_result = None
    stamp_result = None
    signature_result = None
    checkbox_results = []

    required_elements = config.get("required_elements", {})
    required_checkbox_labels = [
        c["label"] for c in config.get("checkboxes", []) if c.get("required")
    ]

    try:
        for page_index, page_image in enumerate(pages, start=1):
            display_bgr, processed_binary = preprocess_page(page_image)

            try:
                ocr_blocks = ocr_engine.run(display_bgr)
            except OCRProcessingError as exc:
                st.warning(f"'{filename}' page {page_index}: OCR failed ({exc}). Skipping OCR-based checks for this page.")
                logger.error("OCR failed for %s page %s: %s", filename, page_index, exc)
                ocr_blocks = []

            all_blocks_by_page.append(ocr_blocks)
            lines = group_text_lines(ocr_blocks)
            all_lines_by_page.append(lines)
            combined_ocr_text_parts.append(get_full_text(ocr_blocks))

            page_gray_real = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2GRAY)

            # --- Photo detection (first page only, typically where the photo is) ---
            if required_elements.get("student_photo", {}).get("required") and photo_result is None:
                regions = find_non_text_regions(
                    display_bgr.shape[:2], ocr_blocks, page_gray=page_gray_real
                )
                photo_cfg = required_elements["student_photo"]
                photo_result = detect_photo_region(
                    page_gray_real,
                    regions,
                    min_width=int(photo_cfg.get("minimum_width", 100)),
                    min_height=int(photo_cfg.get("minimum_height", 100)),
                    aspect_ratio_min=float(photo_cfg.get("expected_aspect_ratio_min", 0.6)),
                    aspect_ratio_max=float(photo_cfg.get("expected_aspect_ratio_max", 1.2)),
                )

            # --- Stamp detection ---
            if required_elements.get("stamp", {}).get("required") and stamp_reference is not None and stamp_result is None:
                ref_cv = pil_to_cv2(stamp_reference)
                candidate = detect_stamp(
                    display_bgr,
                    ref_cv,
                    similarity_threshold=float(
                        required_elements["stamp"].get("similarity_threshold", 0.70)
                    ),
                )
                if stamp_result is None or candidate.similarity > stamp_result.similarity:
                    stamp_result = candidate

            # --- Signature detection ---
            if required_elements.get("signature", {}).get("required") and signature_result is None:
                # Best-effort: look for a region near a "Signature" heading;
                # otherwise fall back to the bottom-right quadrant of the page.
                sig_bbox = _locate_signature_region(display_bgr.shape, all_heading_results, page_index)
                candidate_sig = analyze_signature_region(page_gray_real, sig_bbox)
                signature_result = candidate_sig

            # --- Checkbox detection ---
            page_checkboxes = detect_checkboxes(page_gray_real, ocr_blocks)
            checkbox_results.extend(page_checkboxes)

            annotated_pages.append(cv2_to_pil(display_bgr))

        # --- Heading validation across all pages ---
        heading_results = validate_headings(
            required_headings=config.get("required_headings", []),
            lines_by_page=all_lines_by_page,
            blocks_by_page=all_blocks_by_page,
            heading_match_threshold=float(config.get("heading_match_threshold", 0.75)),
            min_ocr_confidence=float(config.get("minimum_ocr_confidence", 0.60)),
        )

        result = compute_score(
            heading_results=heading_results,
            photo_result=photo_result,
            stamp_result=stamp_result,
            signature_result=signature_result,
            checkbox_results=checkbox_results,
            required_elements=required_elements,
            required_checkbox_labels=required_checkbox_labels,
            scoring_config=config.get("scoring", {}),
        )
        result.filename = filename
        result.pages = len(pages)
        result.ocr_text = "\n\n".join(combined_ocr_text_parts)
        result.processing_duration_seconds = round(time.time() - start_time, 2)

        # Re-draw annotations now that heading_results is final (only for page 1
        # display simplicity; each page gets headings that belong to it).
        final_annotated = []
        for page_index, page_pil in enumerate(annotated_pages, start=1):
            page_bgr = pil_to_cv2(page_pil)
            annotated_bgr = draw_annotations(
                page_bgr,
                heading_results,
                photo_result if page_index == 1 else None,
                stamp_result if page_index == 1 else None,
                signature_result if page_index == 1 else None,
                [c for c in checkbox_results],
                page_index,
            )
            final_annotated.append(cv2_to_pil(annotated_bgr))

        return result, final_annotated

    except Exception as exc:  # noqa: BLE001
        st.error(
            f"'{filename}': processing failed unexpectedly. The other files will "
            "still be processed. See logs for technical details."
        )
        logger.error("Pipeline failure for %s: %s\n%s", filename, exc, traceback.format_exc())
        return None, annotated_pages


def _locate_signature_region(page_shape, heading_results, current_page: int):
    """Best-effort signature region: near a matched 'Signature'/'Declaration'
    heading if found, otherwise the bottom-right quadrant of the page.
    """
    height, width = page_shape[:2]
    for hr in heading_results:
        if hr.page == current_page and hr.bbox and (
            "signature" in hr.heading.lower() or "declaration" in hr.heading.lower()
        ):
            x1, y1, x2, y2 = hr.bbox
            region_x1 = x1
            region_y1 = y2
            region_x2 = min(width, x2 + 250)
            region_y2 = min(height, y2 + 120)
            if region_x2 > region_x1 and region_y2 > region_y1:
                return (region_x1, region_y1, region_x2, region_y2)

    # Fallback: bottom-right quadrant.
    return (int(width * 0.55), int(height * 0.80), width, height)


# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
def render_upload_and_results() -> None:
    st.title("📋 Form Analyzer")
    st.caption("Select a saved format, upload forms, and validate them against that format's rules.")

    selected = render_template_selector()
    if not selected:
        st.info("Create a format first in Template Manager.")
        return

    config = selected["config"]
    slug = selected["slug"]
    refs = get_template_references(slug)
    st.caption(f"Using **{config.get('form_name', slug)}** · {len(refs)} saved reference image(s)")

    uploaded_files = st.file_uploader(
        "Upload form(s) — PDF, PNG, JPG, JPEG",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"analysis_upload_{slug}",
    )

    if uploaded_files:
        with st.expander("Preview uploaded files", expanded=False):
            cols = st.columns(min(4, len(uploaded_files)))
            for i, uf in enumerate(uploaded_files):
                with cols[i % len(cols)]:
                    st.write(uf.name)
                    if uf.type in ("image/png", "image/jpeg"):
                        st.image(uf, width=150)
                    else:
                        st.caption("PDF — preview after analysis")

    analyze_clicked = st.button("Analyze Forms", type="primary", disabled=not uploaded_files)

    if analyze_clicked and uploaded_files:
        # Use the first saved reference as the optional stamp reference for legacy stamp matching.
        stamp_reference = None
        logo_reference = None
        for ref in refs:
            if ref.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                img = load_reference_image(ref)
                if img is not None:
                    # Reference images are whole forms, not stamp/logo crops, so do not treat them as stamp/logo.
                    pass

        results: List[FormValidationResult] = []
        annotated_map: Dict[str, List[Image.Image]] = {}
        progress = st.progress(0.0, text="Starting analysis...")
        for i, uf in enumerate(uploaded_files):
            progress.progress(i / len(uploaded_files), text=f"Analyzing {uf.name}...")
            try:
                file_bytes = uf.getvalue()
            except Exception as exc:
                st.error(f"Could not read uploaded file '{uf.name}': {exc}")
                continue
            result, annotated_pages = analyze_single_file(uf.name, file_bytes, config, stamp_reference, logo_reference)
            if result is not None:
                results.append(result)
                annotated_map[uf.name] = annotated_pages
                st.session_state.db.insert_result(
                    filename=result.filename,
                    status=result.overall_status,
                    score=result.score,
                    ocr_text=result.ocr_text,
                    missing_items=result.missing_headings + result.missing_elements,
                    warnings=result.warnings,
                    result_dict=result.to_dict(),
                    processing_duration_seconds=result.processing_duration_seconds,
                )
        progress.progress(1.0, text="Analysis complete.")
        st.session_state.results = results
        st.session_state.annotated_images = annotated_map
        st.session_state.last_analysis_template = config.get("form_name", slug)

    render_results()


def render_results() -> None:
    results: List[FormValidationResult] = st.session_state.get("results", [])
    if not results:
        return

    st.header("Results")

    from modules.report_generator import results_to_dataframe

    df = results_to_dataframe(results)
    st.dataframe(df, width="stretch")

    col_csv, col_json = st.columns(2)
    with col_csv:
        csv_path = export_csv(results, OUTPUT_DIR / "results.csv")
        st.download_button(
            "Download CSV",
            data=csv_path.read_bytes(),
            file_name="form_analysis_results.csv",
            mime="text/csv",
        )
    with col_json:
        json_path = export_json(results, OUTPUT_DIR / "results.json")
        st.download_button(
            "Download JSON",
            data=json_path.read_bytes(),
            file_name="form_analysis_results.json",
            mime="application/json",
        )

    st.subheader("Per-form detail")
    for result in results:
        status_emoji = "✅" if result.overall_status == "PASS" else "❌"
        with st.expander(f"{status_emoji} {result.filename} — {result.overall_status} ({result.score}%)"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Score", f"{result.score}%")
            c2.metric("Pages", result.pages)
            c3.metric("Processing time", f"{result.processing_duration_seconds}s")

            if result.missing_headings:
                st.write("**Missing headings:**", ", ".join(result.missing_headings))
            if result.missing_elements:
                st.write("**Missing elements:**", ", ".join(result.missing_elements))
            if result.low_confidence_items:
                st.write("**Low-confidence items:**", ", ".join(result.low_confidence_items))
            if result.warnings:
                st.write("**Warnings:**")
                for w in result.warnings:
                    st.write(f"- {w}")

            annotated_pages = st.session_state.annotated_images.get(result.filename, [])
            if annotated_pages:
                st.write("**Annotated pages** (green=found, red=missing, orange=low confidence, blue=info)")
                for idx, page_img in enumerate(annotated_pages, start=1):
                    st.image(page_img, caption=f"Page {idx}", width="stretch")
                    buf = io.BytesIO()
                    page_img.save(buf, format="PNG")
                    st.download_button(
                        f"Download annotated page {idx}",
                        data=buf.getvalue(),
                        file_name=f"{Path(result.filename).stem}_page{idx}_annotated.png",
                        mime="image/png",
                        key=f"dl_{result.filename}_{idx}",
                    )

            with st.expander("OCR extracted text"):
                st.text(result.ocr_text or "(no text extracted)")

            with st.expander("Heading match details"):
                st.json(result.heading_details)

            with st.expander("Checkbox detection details"):
                st.json(result.checkbox_details)


def render_history_page() -> None:
    st.header("Analysis History")
    db: FormDatabase = st.session_state.db
    history = db.fetch_history()

    if not history:
        st.info("No analysis history yet.")
        return

    st.dataframe(pd.DataFrame(history), width="stretch")

    selected_id = st.number_input("View full detail for result ID", min_value=0, step=1, value=0)
    if selected_id:
        detail = db.fetch_result_detail(int(selected_id))
        if detail:
            st.json(detail)
        else:
            st.warning("No record found with that ID.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    init_session_state()

    tab_analyze, tab_templates, tab_config, tab_history = st.tabs([
        "Analyze Forms", "Template Manager", "Template Configuration", "History"
    ])
    with tab_analyze:
        render_upload_and_results()
    with tab_templates:
        render_template_manager()
    with tab_config:
        render_template_configuration()
    with tab_history:
        render_history_page()


if __name__ == "__main__":
    main()