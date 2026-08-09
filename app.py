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
    build_result,
    draw_annotations,
    export_csv,
    export_json,
)
from modules.signature_detector import analyze_signature_region
from modules.stamp_detector import detect_stamp
from modules.template_manager import TemplateManager, DEFAULT_TEMPLATE
from modules.template_analyzer import learn_template_profile
from modules.template_validator import validate_document, GenericValidation

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
        st.session_state.template_manager = TemplateManager(
            TEMPLATE_ROOT, DEFAULT_CONFIG_PATH)
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
    info = next((x for x in manager.list_templates()
                if x["slug"] == slug), None)
    return {"slug": slug, "config": config, "info": info}


def validate_template_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate the generic learned-template schema."""
    errors: List[str] = []
    if not isinstance(config, dict):
        return False, ["Template configuration must be a JSON object."]

    if "structure" not in config:
        errors.append("Missing 'structure' section.")
    elif not isinstance(config.get("structure", {}).get("anchors", []), list):
        errors.append("'structure.anchors' must be a list.")

    if "visual_checks" not in config:
        errors.append("Missing 'visual_checks' section.")

    if "validation" not in config:
        errors.append("Missing 'validation' section.")

    if "scoring" not in config:
        errors.append("Missing 'scoring' section.")

    return len(errors) == 0, errors


def load_reference_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def get_template_references(slug: str) -> List[Path]:
    return st.session_state.template_manager.reference_files(slug)


def render_template_selector(
    widget_key: str = "template_selector"
) -> Optional[Dict[str, Any]]:
    manager: TemplateManager = st.session_state.template_manager
    templates = manager.list_templates()

    if not templates:
        st.warning("No saved formats yet. Create one in Template Manager.")
        return None

    labels = [x["name"] for x in templates]
    current_slug = st.session_state.get("selected_template")
    current_index = next(
        (
            i
            for i, x in enumerate(templates)
            if x["slug"] == current_slug
        ),
        0,
    )

    selected_name = st.selectbox(
        "Format to check", labels, index=current_index, key=widget_key)

    selected = templates[labels.index(selected_name)]
    st.session_state.selected_template = selected["slug"]

    return {
        "slug": selected["slug"],
        "config": manager.load(selected["slug"]),
        "info": selected
    }


def render_template_manager() -> None:
    st.header("Template Manager")
    st.caption(
        "Create and save reusable document formats. "
        "Each format has its own rules and reference images."
    )

    manager: TemplateManager = st.session_state.template_manager
    templates = manager.list_templates()

    # ------------------------------------------------------------------
    # Saved formats
    # ------------------------------------------------------------------
    if templates:
        st.subheader("Saved formats")

        rows = [
            {
                "Format": x["name"],
                "Reference images": x["reference_count"],
                "Folder": x["slug"],
            }
            for x in templates
        ]

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

    # ------------------------------------------------------------------
    # Create new format
    # ------------------------------------------------------------------
    st.subheader("Create a new format")

    with st.form("create_template_form"):
        new_name = st.text_input(
            "Format name",
            placeholder="e.g. Exam A, Exam B, CSR 2027",
        )

        base_options = ["Blank configuration"] + [
            x["name"] for x in templates
        ]

        base_choice = st.selectbox(
            "Start configuration from",
            base_options,
        )

        reference_files = st.file_uploader(
            "Upload known-good template/reference images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="new_template_refs",
        )

        create_clicked = st.form_submit_button(
            "Save New Format",
            type="primary",
        )

    if create_clicked:
        if not new_name.strip():
            st.error("Enter a format name.")

        elif manager.exists(new_name):
            st.error("A format with this name already exists.")

        else:
            try:
                # ------------------------------------------------------
                # Get base configuration
                # ------------------------------------------------------
                if base_choice == "Blank configuration":
                    base_config = json.loads(
                        json.dumps(DEFAULT_TEMPLATE)
                    )
                    base_config["form_name"] = new_name.strip()

                else:
                    base_slug = next(
                        x["slug"]
                        for x in templates
                        if x["name"] == base_choice
                    )

                    base_config = manager.load(base_slug)

                # ------------------------------------------------------
                # Create template
                # ------------------------------------------------------
                slug = manager.create(
                    new_name.strip(),
                    config=base_config,
                )

                # ------------------------------------------------------
                # Save reference images
                # ------------------------------------------------------
                if reference_files:
                    manager.add_references(
                        slug,
                        reference_files,
                    )

                    # --------------------------------------------------
                    # Automatically learn template
                    # --------------------------------------------------
                    try:
                        manager.learn_from_references(slug)

                    except Exception as learn_exc:
                        logger.warning(
                            "Initial template learning failed for %s: %s",
                            slug,
                            learn_exc,
                        )

                        st.warning(
                            "Template was created, but automatic "
                            "learning failed. You can use "
                            "'Learn / Rebuild Template From References' "
                            "below."
                        )

                st.session_state.selected_template = slug

                st.success(
                    f"Format '{new_name.strip()}' saved successfully."
                )

                st.rerun()

            except Exception as exc:
                logger.error(
                    "Template creation failed: %s\n%s",
                    exc,
                    traceback.format_exc(),
                )

                st.error(
                    f"Could not create format: {exc}"
                )

    # ------------------------------------------------------------------
    # Refresh template list after creation
    # ------------------------------------------------------------------
    templates = manager.list_templates()

    if not templates:
        return

    # ------------------------------------------------------------------
    # Manage existing format
    # ------------------------------------------------------------------
    st.subheader("Manage an existing format")

    selected_name = st.selectbox(
        "Format",
        [x["name"] for x in templates],
        key="manager_selected",
    )

    selected = next(
        x for x in templates
        if x["name"] == selected_name
    )

    slug = selected["slug"]
    config = manager.load(slug)

    st.divider()

    # ==================================================================
    # THREE COLUMN MANAGEMENT AREA
    # ==================================================================

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------------------
    # Rename
    # ------------------------------------------------------------------
    with col1:
        st.write("Rename format")

        new_name = st.text_input(
            "New name",
            value=config.get(
                "form_name",
                selected_name,
            ),
            key="rename_name",
        )

        if st.button(
            "Rename",
            key="rename_btn",
        ):
            if not new_name.strip():
                st.error("Enter a format name.")

            elif (
                new_name.strip().lower()
                != selected_name.lower()
                and manager.exists(new_name.strip())
            ):
                st.error(
                    "A format with this name already exists."
                )

            else:
                try:
                    new_slug = manager.rename(
                        slug,
                        new_name.strip(),
                    )

                    st.session_state.selected_template = new_slug

                    st.success(
                        "Format renamed successfully."
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(str(exc))

    # ------------------------------------------------------------------
    # Add references
    # ------------------------------------------------------------------
    with col2:
        st.write("Reference images")

        more_refs = st.file_uploader(
            "Add reference images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="more_refs",
        )

        if st.button(
            "Add References",
            key="add_refs_btn",
        ):
            if not more_refs:
                st.warning(
                    "Select at least one image."
                )

            else:
                try:
                    count = manager.add_references(
                        slug,
                        more_refs,
                    )

                    st.success(
                        f"Added {count} reference image(s)."
                    )

                    st.rerun()

                except Exception as exc:
                    logger.error(
                        "Failed to add references: %s\n%s",
                        exc,
                        traceback.format_exc(),
                    )

                    st.error(
                        f"Could not add references: {exc}"
                    )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    with col3:
        st.write("Danger zone")

        if st.button(
            "🗑️ Delete Format",
            key="delete_template_btn",
            type="secondary",
        ):
            # Don't allow the application to have zero templates.
            if len(templates) == 1:
                st.error(
                    "You cannot delete the only remaining format. "
                    "Create another format first."
                )

            else:
                try:
                    manager.delete(slug)

                    # Get remaining templates
                    remaining = manager.list_templates()

                    # Select another template automatically
                    if remaining:
                        st.session_state.selected_template = (
                            remaining[0]["slug"]
                        )
                    else:
                        st.session_state.selected_template = None

                    st.success(
                        f"Format '{selected_name}' deleted successfully."
                    )

                    st.rerun()

                except Exception as exc:
                    logger.error(
                        "Failed to delete template %s: %s\n%s",
                        slug,
                        exc,
                        traceback.format_exc(),
                    )

                    st.error(
                        f"Failed to delete template: {exc}"
                    )

    # ------------------------------------------------------------------
    # Reference images / learning
    # ------------------------------------------------------------------
    refs = manager.reference_files(slug)

    st.divider()

    st.write(
        f"**{len(refs)} reference image(s)**"
    )

    if len(refs) < 3:
        st.info(
            "Add at least 3 known-good reference images. "
            "5+ is recommended."
        )

    # ------------------------------------------------------------------
    # Learn / rebuild template
    # ------------------------------------------------------------------
    if st.button(
        "Learn / Rebuild Template From References",
        key=f"learn_{slug}",
        type="primary",
    ):
        try:
            if not refs:
                st.error(
                    "Add at least one reference image first."
                )

            else:
                with st.spinner(
                    "Analyzing reference forms and "
                    "learning their common structure..."
                ):
                    learned = manager.learn_from_references(
                        slug
                    )

                st.success(
                    f"Template learned from "
                    f"{learned.get('reference_count', len(refs))} "
                    f"reference image(s). "
                    f"Found "
                    f"{len(learned.get('structure', {}).get('anchors', []))} "
                    f"common structural anchors."
                )

                st.rerun()

        except Exception as exc:
            logger.error(
                "Template learning failed for %s: %s\n%s",
                slug,
                exc,
                traceback.format_exc(),
            )

            st.error(
                f"Template learning failed: {exc}"
            )

    # ------------------------------------------------------------------
    # Show saved reference images
    # ------------------------------------------------------------------
    if refs:
        st.subheader("Saved reference images")

        cols = st.columns(
            min(5, len(refs))
        )

        for i, ref in enumerate(refs):
            with cols[i % len(cols)]:
                img = load_reference_image(ref)

                if img is not None:
                    st.image(
                        img,
                        caption=ref.name,
                        width=150,
                    )


def render_template_configuration() -> None:
    st.header("Template Configuration")
    st.caption(
        "Rules are stored per format. You can change them without editing Python code.")
    selected = render_template_selector(
        widget_key="configuration_template_selector")
    if not selected:
        return
    manager: TemplateManager = st.session_state.template_manager
    slug = selected["slug"]
    config = selected["config"]

    st.info(f"Editing: **{config.get('form_name', slug)}**")
    edited = st.text_area("Template JSON", value=json.dumps(
        config, indent=2, ensure_ascii=False), height=600)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Configuration", type="primary"):
            try:
                parsed = json.loads(edited)
                if not isinstance(parsed, dict):
                    raise ValueError(
                        "Configuration root must be a JSON object.")
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
            manager.save(slug, {**DEFAULT_TEMPLATE,
                         "form_name": config.get("form_name", slug)})
            st.success("Configuration reset.")
            st.rerun()

# --------------------------------------------------------------------------
# Core per-file analysis pipeline
# --------------------------------------------------------------------------


def analyze_single_file(
    filename: str,
    file_bytes: bytes,
    config: Dict[str, Any],
) -> tuple[Optional[FormValidationResult], List[Image.Image]]:
    """Analyze one document against the selected generic template profile."""
    start_time = time.time()
    annotated_pages: List[Image.Image] = []

    try:
        pages = load_file_as_images(filename, file_bytes)
    except Exception as exc:
        logger.error("File load failed for %s: %s", filename, exc)
        st.error(f"'{filename}': could not load file: {exc}")
        return None, []

    if not pages:
        st.error(f"'{filename}': no pages found.")
        return None, []

    ocr_engine = OCREngine.get_instance()
    all_blocks_by_page = []
    combined_text = []

    try:
        first_page_gray = None

        for page_index, page_image in enumerate(pages, start=1):
            display_bgr, _ = preprocess_page(page_image)

            if page_index == 1:
                first_page_gray = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2GRAY)

            try:
                blocks = ocr_engine.run(display_bgr)
            except OCRProcessingError as exc:
                logger.error("OCR failed for %s page %s: %s",
                             filename, page_index, exc)
                blocks = []
                st.warning(f"'{filename}' page {page_index}: OCR failed.")

            all_blocks_by_page.append(blocks)
            combined_text.append(get_full_text(blocks))
            annotated_pages.append(cv2_to_pil(display_bgr))

        if first_page_gray is None:
            raise ValueError("Could not process first page.")

        first_blocks = all_blocks_by_page[0] if all_blocks_by_page else []

        validation = validate_document(
            first_page_gray,
            first_blocks,
            config,
        )

        # Adapt generic validation to the existing report structure.
        class _Classification:
            document_type = validation.document_type
            confidence = validation.classification_confidence
            passed = validation.classification_confidence >= float(
                config.get("validation", {}).get("require_anchor_ratio", 0.70)
            )
            matched_markers = validation.matched_anchors
            missing_markers = validation.missing_anchors
            details = f"Matched {len(validation.matched_anchors)} template anchors."

        class _Compat:
            classification = _Classification()
            checks = validation.checks
            missing_items = validation.missing_items
            warnings = validation.warnings
            format_passed = validation.structure_score >= float(
                config.get("scoring", {}).get("pass_score_threshold", 75)
            )
            format_score = validation.structure_score

            def to_dict(self):
                return validation.to_dict()

        compat = _Compat()

        result = build_result(
            filename=filename,
            pages=len(pages),
            validation=compat,
            ocr_text="\n\n".join(combined_text),
            processing_duration_seconds=round(time.time() - start_time, 2),
            scoring_config={
                "classification_weight": 0,
                "format_weight": config.get("scoring", {}).get("structure_weight", 70),
                "visual_weight": config.get("scoring", {}).get("visual_weight", 30),
                "pass_score_threshold": config.get("scoring", {}).get("pass_score_threshold", 75),
                "require_all_mandatory_for_pass": False,
            },
        )

        # Use the generic score as the authoritative score.
        result.score = validation.overall_score
        result.max_score = 100.0
        result.overall_status = (
            "PASS"
            if validation.overall_score >= float(
                config.get("scoring", {}).get("pass_score_threshold", 75)
            )
            and (
                validation.classification_confidence
                >= float(config.get("validation", {}).get("require_anchor_ratio", 0.70))
                if validation.matched_anchors
                else False
            )
            else "FAIL"
        )

        result.document_type = validation.document_type
        result.document_confidence = round(
            validation.classification_confidence * 100, 2)
        result.format_score = validation.structure_score
        result.format_status = "PASS" if validation.structure_score >= float(
            config.get("scoring", {}).get("pass_score_threshold", 75)
        ) else "FAIL"
        result.required_fields_passed = len(validation.matched_anchors)
        result.required_fields_total = len(
            config.get("structure", {}).get("anchors", [])
        )
        result.missing_elements = validation.missing_items
        result.warnings = validation.warnings
        result.ocr_text = "\n\n".join(combined_text)
        result.validation_details = validation.to_dict()
        result.processing_duration_seconds = round(time.time() - start_time, 2)

        # Visual annotation.
        final_pages = []
        for page_no, page_img in enumerate(annotated_pages, start=1):
            if page_no == 1:
                bgr = pil_to_cv2(page_img)
                # Draw generic check boxes.
                for check in validation.checks:
                    if check.bbox:
                        color = (0, 170, 0) if check.passed else (0, 0, 220)
                        x1, y1, x2, y2 = check.bbox
                        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            bgr,
                            check.label[:30],
                            (x1, max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                final_pages.append(cv2_to_pil(bgr))
            else:
                final_pages.append(page_img)

        return result, final_pages

    except Exception as exc:
        logger.error(
            "Pipeline failure for %s: %s\n%s",
            filename,
            exc,
            traceback.format_exc(),
        )
        st.error(f"'{filename}': processing failed: {exc}")
        return None, annotated_pages


# def _locate_signature_region(page_shape, heading_results, current_page: int):
#     """Best-effort signature region: near a matched 'Signature'/'Declaration'
#     heading if found, otherwise the bottom-right quadrant of the page.
#     """
#     height, width = page_shape[:2]
#     for hr in heading_results:
#         if hr.page == current_page and hr.bbox and (
#             "signature" in hr.heading.lower() or "declaration" in hr.heading.lower()
#         ):
#             x1, y1, x2, y2 = hr.bbox
#             region_x1 = x1
#             region_y1 = y2
#             region_x2 = min(width, x2 + 250)
#             region_y2 = min(height, y2 + 120)
#             if region_x2 > region_x1 and region_y2 > region_y1:
#                 return (region_x1, region_y1, region_x2, region_y2)

#     # Fallback: bottom-right quadrant.
#     return (int(width * 0.55), int(height * 0.80), width, height)


# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
def render_upload_and_results() -> None:
    st.title("📋 Form Analyzer")
    st.caption(
        "Select a saved format, upload forms, and validate them against that format's rules.")

    selected = render_template_selector(
        widget_key="analysis_template_selector")
    if not selected:
        st.info("Create a format first in Template Manager.")
        return

    config = selected["config"]

    config_valid, config_errors = validate_template_config(config)

    if not config_valid:

        st.error(
            f"Template '{selected['info']['name']}' has an invalid "
            "configuration."
        )

        st.warning(
            "This format has not learned its structure yet. Add reference images and use Learn Template."
        )

        with st.expander("Configuration errors"):
            for error in config_errors:
                st.write(f"- {error}")

        return

    config = selected["config"]
    slug = selected["slug"]
    refs = get_template_references(slug)
    st.caption(
        f"Using **{config.get('form_name', slug)}** · {len(refs)} saved reference image(s)")

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

    analyze_clicked = st.button(
        "Analyze Forms", type="primary", disabled=not uploaded_files)

    if analyze_clicked and uploaded_files:
        # Use the first saved reference as the optional stamp reference for legacy stamp matching.
        # stamp_reference = None
        # logo_reference = None
        # for ref in refs:
        #     if ref.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        #         img = load_reference_image(ref)
        #         if img is not None:
        #             # Reference images are whole forms, not stamp/logo crops, so do not treat them as stamp/logo.
        #             pass

        results: List[FormValidationResult] = []
        annotated_map: Dict[str, List[Image.Image]] = {}
        progress = st.progress(0.0, text="Starting analysis...")
        for i, uf in enumerate(uploaded_files):
            progress.progress(
                (i + 1) / len(uploaded_files),
                text=f"Analyzing {uf.name}..."
            )
            try:
                file_bytes = uf.getvalue()
            except Exception as exc:
                st.error(f"Could not read uploaded file '{uf.name}': {exc}")
                continue
            result, annotated_pages = analyze_single_file(
                uf.name, file_bytes, config)
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
            c3.metric("Processing time",
                      f"{result.processing_duration_seconds}s")

            if result.missing_headings:
                st.write("**Missing headings:**",
                         ", ".join(result.missing_headings))
            if result.missing_elements:
                st.write("**Missing elements:**",
                         ", ".join(result.missing_elements))
            if result.low_confidence_items:
                st.write("**Low-confidence items:**",
                         ", ".join(result.low_confidence_items))
            if result.warnings:
                st.write("**Warnings:**")
                for w in result.warnings:
                    st.write(f"- {w}")

            annotated_pages = st.session_state.annotated_images.get(
                result.filename, [])
            if annotated_pages:
                st.write(
                    "**Annotated pages** (green=found, red=missing, orange=low confidence, blue=info)")
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

            with st.expander("Validation details"):
                st.json(result.validation_details)

            with st.expander("Individual checks"):
                st.json(result.checks)


def render_history_page() -> None:
    st.header("Analysis History")
    db: FormDatabase = st.session_state.db
    history = db.fetch_history()

    if not history:
        st.info("No analysis history yet.")
        return

    st.dataframe(pd.DataFrame(history), width="stretch")

    selected_id = st.number_input(
        "View full detail for result ID", min_value=0, step=1, value=0)
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
