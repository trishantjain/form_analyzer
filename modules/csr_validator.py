"""Backward-compatible wrapper.

The application now uses modules.template_validator directly.
This file remains so older imports do not break.
"""

from modules.template_validator import validate_document, GenericValidation


def validate_csr_page(page_gray, blocks, config):
    """Compatibility alias for older code paths."""
    return validate_document(page_gray, blocks, config)