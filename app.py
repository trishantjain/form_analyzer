"""Professional, config-driven CSR Form Analyzer.

Run with:
    streamlit run app.py

The application deliberately separates:
1. Document classification (CSR vs other document)
2. CSR format/field validation
3. Signature / stamp / seal region checks
4. Optional reference-template layout matching

All check parameters are editable in config/form_template.json and can also
be changed from the Configuration tab without modifying Python code.
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
import pandas as pd
import streamlit as st
from PIL import Image

from modules.csr_validator import CSRValidation, validate_csr_page
from modules.database import FormDatabase
from modules.image_preprocessing import cv2_to_pil, pil_to_cv2, preprocess_page
from modules.ocr_engine import OCREngine, OCRProcessingError, get_full_text
from modules.pdf_utils import PDFProcessingError, load_file_as_images
from modules.report_generator import (
    FormValidationResult,
    build_result,
    draw_annotations,
    export_csv,
    export_json,
    results_to_dataframe,
)
from modules.template_matcher import compare_to_references

LOG_DIR = Path("output")
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("output")
DB_PATH = OUTPUT_DIR / "form_analyzer.db"
CONFIG_PATH = Path("config/form_template.json")
TEMPLATE_DIR = Path("templates/csr")
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "form_analyzer.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("form_analyzer.app")

st.set_page_config(page_title="CSR Form Analyzer", page_icon="📋", layout="wide")


def load_config() -> Dict[str, Any]:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not load {CONFIG_PATH}: {exc}")
        st.stop()


def save_config(config: Dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def init_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "results" not in st.session_state:
        st.session_state.results = []
    if "annotated_images" not in st.session_state:
        st.session_state.annotated_images = {}
    if "db" not in st.session_state:
        st.session_state.db = FormDatabase(DB_PATH)
    if "template_references" not in st.session_state:
        st.session_state.template_references = load_reference_images()


def load_reference_images() -> List[tuple[str, Image.Image]]:
    refs = []
    for path in sorted(TEMPLATE_DIR.glob("*.png")) + sorted(TEMPLATE_DIR.glob("*.jpg")) + sorted(TEMPLATE_DIR.glob("*.jpeg")):
        try:
            refs.append((path.name, Image.open(path).convert("RGB")))
        except Exception:
            logger.exception("Could not load reference image %s", path)
    return refs


def render_configuration() -> None:
    config = st.session_state.config
    st.header("⚙️ Configuration")
    st.caption("All validation rules are configuration-driven. You can edit and save them here or directly in config/form_template.json.")

    c1, c2 = st.columns(2)
    with c1:
        config["form_name"] = st.text_input("Form name", config.get("form_name", "CSR Form"))
        cls = config.setdefault("document_classification", {})
        cls["marker_match_threshold"] = st.slider("Document marker match threshold", 0.40, 1.0, float(cls.get("marker_match_threshold", 0.72)), 0.01)
        cls["minimum_confidence"] = st.slider("Minimum CSR classification confidence", 0.0, 1.0, float(cls.get("minimum_confidence", 0.60)), 0.01)
        cls["minimum_markers"] = st.number_input("Minimum matched CSR markers", 1, 50, int(cls.get("minimum_markers", 5)))

    with c2:
        validation = config.setdefault("validation", {})
        validation["field_match_threshold"] = st.slider("Field/section OCR match threshold", 0.40, 1.0, float(validation.get("field_match_threshold", 0.72)), 0.01)
        fmt = config.setdefault("format_checks", {})
        fmt["pass_score_threshold"] = st.slider("Minimum format score", 0, 100, int(fmt.get("pass_score_threshold", 90)))
        scoring = config.setdefault("scoring", {})
        scoring["classification_weight"] = st.number_input("Classification weight", 0, 100, int(scoring.get("classification_weight", 20)))
        scoring["format_weight"] = st.number_input("Format weight", 0, 100, int(scoring.get("format_weight", 60)))
        scoring["visual_weight"] = st.number_input("Signature/stamp weight", 0, 100, int(scoring.get("visual_weight", 20)))
        scoring["pass_score_threshold"] = st.slider("Final PASS score", 0, 100, int(scoring.get("pass_score_threshold", 85)))
        scoring["require_all_mandatory_for_pass"] = st.checkbox("Require all mandatory checks", value=bool(scoring.get("require_all_mandatory_for_pass", True)))

    st.subheader("Document classification markers")
    st.write("Add/remove aliases in JSON for OCR variations. A marker is matched if any alias reaches the configured threshold.")
    st.json(config.get("document_classification", {}).get("required_markers", []))

    st.subheader("Required sections and fields")
    st.json(config.get("format_checks", {}).get("required_sections", []))
    st.json(config.get("format_checks", {}).get("required_fields", []))

    st.subheader("Signature / stamp / seal regions")
    st.caption("Regions use normalized coordinates: x/y/width/height are 0.0–1.0 of the page. This makes the rules independent of image resolution.")
    st.json(config.get("format_checks", {}).get("visual_checks", []))

    st.subheader("Advanced JSON editor")
    json_text = st.text_area("Edit complete configuration", value=json.dumps(config, indent=2), height=520, key="config_json_editor")
    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("💾 Save configuration", type="primary"):
            try:
                parsed = json.loads(json_text)
                save_config(parsed)
                st.session_state.config = parsed
                st.success(f"Saved {CONFIG_PATH}")
                st.rerun()
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
    with col_reset:
        if st.button("↩ Reload config from disk"):
            st.session_state.config = load_config()
            st.rerun()


def render_template_references() -> None:
    st.header("🧩 CSR Template References")
    st.caption("Upload known-good CSR pictures. They are stored locally in templates/csr and are used as optional secondary layout references.")
    uploaded = st.file_uploader("Known-good CSR reference images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="csr_refs")
    if uploaded and st.button("Save reference images"):
        for uf in uploaded:
            (TEMPLATE_DIR / uf.name).write_bytes(uf.getvalue())
        st.session_state.template_references = load_reference_images()
        st.success(f"Saved {len(uploaded)} reference image(s).")
        st.rerun()

    if st.session_state.template_references:
        cols = st.columns(min(5, len(st.session_state.template_references)))
        for i, (name, image) in enumerate(st.session_state.template_references):
            with cols[i % len(cols)]:
                st.image(image, caption=name, use_container_width=True)
    else:
        st.info("No reference images found. OCR/structure checks will still work.")


def analyze_single_file(filename: str, file_bytes: bytes, config: Dict[str, Any]) -> tuple[Optional[FormValidationResult], List[Image.Image]]:
    start = time.time()
    annotated_pages: List[Image.Image] = []
    try:
        pages = load_file_as_images(filename, file_bytes)
    except (PDFProcessingError, ValueError) as exc:
        st.error(f"{filename}: {exc}")
        return None, []
    except Exception as exc:
        st.error(f"{filename}: could not load file")
        logger.error("Load error %s: %s\n%s", filename, exc, traceback.format_exc())
        return None, []

    engine = OCREngine.get_instance(config.get("advanced", {}).get("ocr_language", "en"))
    all_blocks = []
    ocr_parts = []
    first_page_bgr = None
    first_page_gray = None

    try:
        for page_index, page in enumerate(pages, start=1):
            bgr, _binary = preprocess_page(page)
            if first_page_bgr is None:
                first_page_bgr = bgr
                first_page_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            try:
                blocks = engine.run(bgr)
            except OCRProcessingError as exc:
                blocks = []
                logger.warning("OCR failed for %s page %s: %s", filename, page_index, exc)
            all_blocks.extend(blocks)
            ocr_parts.append(get_full_text(blocks))
            annotated_pages.append(cv2_to_pil(bgr))

        if first_page_bgr is None or first_page_gray is None:
            return None, []

        validation = validate_csr_page(first_page_gray, all_blocks, config)

        # Optional reference layout signal.
        template_cfg = config.get("template_matching", {})
        refs = [(name, pil_to_cv2(img)) for name, img in st.session_state.template_references]
        if template_cfg.get("enabled", True) and refs:
            match = compare_to_references(first_page_bgr, refs, float(template_cfg.get("threshold", 0.30)))
            validation.warnings.append(
                f"Reference layout similarity: {match.similarity * 100:.1f}% ({match.reference_name or 'none'})."
            )
            validation_dict = validation.to_dict()
            validation_dict["template_match"] = {
                "matched": match.matched,
                "similarity": match.similarity,
                "reference_name": match.reference_name,
                "good_matches": match.good_matches,
                "inlier_ratio": match.inlier_ratio,
            }
        else:
            validation_dict = validation.to_dict()

        result = build_result(
            filename=filename,
            pages=len(pages),
            validation=validation,
            ocr_text="\n\n".join(ocr_parts),
            processing_duration_seconds=round(time.time() - start, 2),
            scoring_config=config.get("scoring", {}),
        )
        result.validation_details = validation_dict

        annotated = []
        for i, page_img in enumerate(annotated_pages, start=1):
            page_bgr = pil_to_cv2(page_img)
            if i == 1:
                page_bgr = draw_annotations(page_bgr, validation, i)
            annotated.append(cv2_to_pil(page_bgr))
        return result, annotated
    except Exception as exc:
        st.error(f"{filename}: processing failed. Other files will continue.")
        logger.error("Pipeline failure for %s: %s\n%s", filename, exc, traceback.format_exc())
        return None, annotated_pages


def render_analysis() -> None:
    st.header("📋 Analyze CSR Forms")
    st.caption("Upload one or hundreds of scanned/photographed forms. Each file is classified first, then validated only if it looks like the configured CSR form.")
    uploaded = st.file_uploader("Upload CSR forms", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
    if not uploaded:
        return

    st.write(f"**{len(uploaded)} file(s) ready for analysis.**")
    if st.button("🚀 Analyze Forms", type="primary"):
        results = []
        annotated_map = {}
        progress = st.progress(0, text="Starting...")
        for i, uf in enumerate(uploaded):
            progress.progress(i / len(uploaded), text=f"Analyzing {uf.name} ({i + 1}/{len(uploaded)})")
            result, annotated = analyze_single_file(uf.name, uf.getvalue(), st.session_state.config)
            if result:
                results.append(result)
                annotated_map[result.filename] = annotated
                st.session_state.db.insert_result(
                    filename=result.filename,
                    status=result.overall_status,
                    score=result.score,
                    ocr_text=result.ocr_text,
                    missing_items=result.missing_elements,
                    warnings=result.warnings,
                    result_dict=result.to_dict(),
                    processing_duration_seconds=result.processing_duration_seconds,
                )
        progress.progress(1.0, text="Analysis complete")
        st.session_state.results = results
        st.session_state.annotated_images = annotated_map

    render_results()


def render_results() -> None:
    results: List[FormValidationResult] = st.session_state.results
    if not results:
        return
    st.subheader("Results")
    df = results_to_dataframe(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    summary = pd.DataFrame([
        {"Metric": "Total files", "Value": len(results)},
        {"Metric": "CSR forms", "Value": sum(r.document_type != "Other Document" for r in results)},
        {"Metric": "PASS", "Value": sum(r.overall_status == "PASS" for r in results)},
        {"Metric": "FAIL", "Value": sum(r.overall_status == "FAIL" for r in results)},
    ])
    st.dataframe(summary, hide_index=True, use_container_width=False)

    c1, c2 = st.columns(2)
    with c1:
        path = export_csv(results, OUTPUT_DIR / "results.csv")
        st.download_button("Download CSV", path.read_bytes(), "csr_analysis_results.csv", "text/csv")
    with c2:
        path = export_json(results, OUTPUT_DIR / "results.json")
        st.download_button("Download JSON", path.read_bytes(), "csr_analysis_results.json", "application/json")

    for r in results:
        emoji = "✅" if r.overall_status == "PASS" else "❌"
        with st.expander(f"{emoji} {r.filename} — {r.overall_status} ({r.score}%)"):
            a, b, c, d = st.columns(4)
            a.metric("Document", r.document_type)
            b.metric("Format", f"{r.format_score}%")
            c.metric("Signature", r.signatures_status)
            d.metric("Stamp / Seal", r.stamps_status)
            st.write(f"**Required structural checks:** {r.required_fields_passed}/{r.required_fields_total}")
            if r.missing_elements:
                st.error("Missing / failed: " + ", ".join(r.missing_elements))
            if r.warnings:
                for warning in r.warnings:
                    st.warning(warning)
            if r.validation_details.get("classification"):
                st.write("**Classification details**")
                st.json(r.validation_details["classification"])
            st.write("**Check details**")
            st.dataframe(pd.DataFrame(r.checks), use_container_width=True, hide_index=True)

            for idx, image in enumerate(st.session_state.annotated_images.get(r.filename, []), start=1):
                st.image(image, caption=f"Annotated page {idx}", use_container_width=True)
                buf = io.BytesIO(); image.save(buf, format="PNG")
                st.download_button(f"Download annotated page {idx}", buf.getvalue(), f"{Path(r.filename).stem}_page{idx}.png", "image/png", key=f"ann_{r.filename}_{idx}")

            with st.expander("OCR text"):
                st.text(r.ocr_text or "No OCR text")
            with st.expander("Full machine-readable validation"):
                st.json(r.to_dict())


def render_history() -> None:
    st.header("🗃️ History")
    history = st.session_state.db.fetch_history()
    if not history:
        st.info("No analysis history yet.")
        return
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    selected = st.number_input("Result ID", min_value=0, step=1, value=0)
    if selected:
        detail = st.session_state.db.fetch_result_detail(int(selected))
        if detail:
            st.json(detail)


def main() -> None:
    init_state()
    st.title("📋 CSR Form Analyzer")
    st.caption("Local, configurable document classification and CSR validation")
    tab1, tab2, tab3, tab4 = st.tabs(["Analyze", "Template References", "Configuration", "History"])
    with tab1:
        render_analysis()
    with tab2:
        render_template_references()
    with tab3:
        render_configuration()
    with tab4:
        render_history()


if __name__ == "__main__":
    main()
