# Generic Multi-Form Analyzer - Updated Architecture

Replace/add these files:

1. app_updated.py -> app.py
2. template_manager_updated.py -> modules/template_manager.py
3. template_analyzer.py -> modules/template_analyzer.py
4. template_validator.py -> modules/template_validator.py

Optional compatibility file:
5. csr_validator_compat.py -> modules/csr_validator.py

The updated app no longer uses CSR-specific classification for normal validation.
Each saved format has its own learned profile.

Workflow:
1. Open Template Manager.
2. Create a format name (MPSC, CSR, Exam A, etc.).
3. Upload 5+ known-good reference images.
4. Save. The app automatically attempts to learn the profile.
5. You can later add more references.
6. Click "Learn / Rebuild Template From References" after adding references.
7. Select the format in Analyze Forms.
8. Upload your test images (100 or more).
9. Analyze Forms.

The learned profile stores common OCR structural anchors and visual signature/stamp regions.
Reference values/handwriting are not treated as exact values.

IMPORTANT:
- The current analyzer learns printed OCR-visible labels. If a form has very poor OCR,
  anchors may be missed; those references should be improved or manually edited in
  Template Configuration.
- Signature/stamp region detection is based on labels containing signature/stamp/seal
  and is intentionally a first generic implementation. It is not handwriting identity
  verification.
- PDFs currently validate the first page for the generic structural/visual checks,
  matching the existing application's first-page behavior.