# CSR Form Analyzer

A local, configurable Streamlit application for checking large batches of photographed/scanned **Centre Sign-Off Report (CSR) – Biometric** forms.

## What it does

For every uploaded file the pipeline is:

1. **Document classification** — CSR form vs Other Document using configurable OCR markers.
2. **Format validation** — checks required sections and field labels, without requiring the actual handwritten values to match the template.
3. **Signature / stamp / seal checks** — checks configurable normalized page regions for ink/markings.
4. **Optional template reference matching** — ORB/homography provides a secondary layout-similarity signal against known-good CSR images.
5. **Final decision** — PASS / FAIL with transparent scores and reasons.

The design deliberately does **not** compare the complete image pixel-for-pixel. Names, dates, candidate counts, handwriting, signatures and stamps can legitimately vary between correct CSR forms.

## Project structure

```text
form_analyzer/
├── app.py
├── config/
│   └── form_template.json       # Main professional configuration file
├── modules/
│   ├── csr_validator.py         # CSR classification + format + visual checks
│   ├── template_matcher.py      # Optional reference layout matcher
│   ├── report_generator.py      # Score, results, annotation, CSV/JSON
│   ├── ocr_engine.py
│   ├── image_preprocessing.py
│   ├── pdf_utils.py
│   └── database.py
├── templates/
│   └── csr/                     # Known-good CSR reference pictures
└── output/
```

## Configuration — no code changes required

The main file is:

```text
config/form_template.json
```

You can also edit the same configuration from the **Configuration** tab in the application.

### 1. Document classification

```json
"document_classification": {
  "marker_match_threshold": 0.72,
  "minimum_confidence": 0.60,
  "minimum_markers": 5,
  "required_markers": [ ... ]
}
```

- `marker_match_threshold`: OCR fuzzy-match threshold.
- `minimum_markers`: minimum number of CSR markers that must be found.
- `required_markers`: add/remove marker groups and aliases for your form.

### 2. Required fields and sections

Under `format_checks`:

```json
"required_sections": [ ... ],
"required_fields": [ ... ]
```

Each item can contain:

```json
{
  "id": "centre_code",
  "label": "Centre Code",
  "aliases": ["Centre Code", "Center Code"],
  "required": true
}
```

This checks that the **field/label exists**. It does not compare the handwritten value with the sample.

### 3. Signature / stamp / seal regions

Visual checks use normalized coordinates:

```json
{
  "id": "venue_rep_signature_stamp",
  "label": "Venue Representative signature & stamp",
  "type": "signature_and_stamp_region",
  "required": true,
  "region": {
    "x": 0.64,
    "y": 0.30,
    "width": 0.34,
    "height": 0.14
  },
  "min_ink_percentage": 1.2
}
```

Coordinates are fractions of the page:

- `x`, `y`: top-left
- `width`, `height`: region size
- all values are normally between `0.0` and `1.0`

This is resolution-independent. If your actual CSR layout changes, adjust these values rather than changing Python code.

**Important:** the current visual detector establishes the presence of ink/markings in the configured region. It does not prove that a stamp is authentic or that a signature belongs to a particular person.

### 4. Scoring

```json
"scoring": {
  "classification_weight": 20,
  "format_weight": 60,
  "visual_weight": 20,
  "pass_score_threshold": 85,
  "require_all_mandatory_for_pass": true
}
```

Adjust these values according to your organization's acceptance criteria.

### 5. Template matching

Put known-good CSR images in:

```text
templates/csr/
```

or upload them from the **Template References** tab.

```json
"template_matching": {
  "enabled": true,
  "threshold": 0.30,
  "warning_only": true
}
```

Template matching is intentionally a **secondary signal** because a completed form contains variable handwriting and values. It should not replace OCR/field validation.

## Running

```bash
python -m venv venv
venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
streamlit run app.py
```

The first PaddleOCR run may download its local model files. Actual analysis is local.

## Recommended production workflow

1. Put 5–10 known-good CSR photographs/scans into `templates/csr/`.
2. Review `config/form_template.json` and mark every field/section as required or optional.
3. Adjust the normalized signature/stamp/seal regions if your camera framing differs.
4. Test 10–20 known-good forms first.
5. Test deliberately broken forms: missing section, wrong document, missing signature, missing stamp/seal.
6. Only then process the 100+ production images.

The CSV/JSON result includes document classification, format result, field counts, signature/stamp status, final score, missing checks, warnings and machine-readable validation details.
