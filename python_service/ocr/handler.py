"""
OCR handler — backend selection and unified interface.

Every other module imports `ocr_page` from here. Which backend runs
is decided once at import time. Nothing outside this file knows whether
it's talking to Tesseract or (if we add it later) PaddleOCR.
"""

import logging
from typing import Callable

from python_service.ocr.tesseract_backend import OcrResult, ocr_page as _tesseract_ocr_page

logger = logging.getLogger(__name__)

# The active backend. Starts as Tesseract; swap in paddle_backend here if needed.
_backend_name: str = "tesseract"
_backend_fn: Callable[[bytes], OcrResult] = _tesseract_ocr_page

logger.info("OCR backend: %s", _backend_name)


def ocr_page(image_bytes: bytes) -> OcrResult:
    """
    OCR a single page.

    Args:
        image_bytes: raw bytes of a PNG/JPEG/TIFF page image.

    Returns:
        OcrResult with per-line confidence, annotated full_text,
        and overall avg_confidence.

    Raises:
        ValueError: if image bytes can't be decoded.
        RuntimeError: if the OCR backend itself errors.
    """
    return _backend_fn(image_bytes)


def get_backend_name() -> str:
    """Returns the name of the active OCR backend ('tesseract' or 'paddle')."""
    return _backend_name
