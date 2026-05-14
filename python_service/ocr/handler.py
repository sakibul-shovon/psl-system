"""
OCR handler — cascade: Tesseract first, TrOCR fallback for handwriting.

Every other module imports `ocr_page` from here.

Two-stage pipeline
──────────────────
Stage 1: Tesseract (always runs first — fast, great on printed/scanned text)
Stage 2: TrOCR    (runs ONLY when Tesseract avg_confidence < 0.35)

Why this split?
  Tesseract on a clean typed scan: avg_confidence ≈ 0.80–0.95
  Tesseract on handwriting:        avg_confidence ≈ 0.05–0.30
  The confidence score is the automatic handwriting detector — no manual
  flagging required.

What about TEXT_LAYER pages?
  Pages where pypdf finds ≥ 20 characters of text never call `ocr_page` at all.
  The cascade only fires for pages that already needed OCR.
  Good PDFs → zero overhead, zero TrOCR downloads triggered.
"""

import logging

from python_service.ocr.tesseract_backend import OcrResult, ocr_page as _tesseract_ocr_page

logger = logging.getLogger(__name__)

# Tesseract confidence threshold below which we escalate to TrOCR.
# Typed scans score ≥ 0.70; handwriting typically scores 0.05–0.30.
# 0.35 is a conservative cut: avoids triggering on slightly blurry scans
# while reliably catching handwriting.
TROCR_FALLBACK_THRESHOLD = 0.35


def ocr_page(image_bytes: bytes) -> OcrResult:
    """
    OCR a single page with automatic handwriting detection.

    Flow:
      1. Run Tesseract.
      2. avg_confidence >= 0.35 → return Tesseract result immediately.
      3. avg_confidence <  0.35 → likely handwriting → try TrOCR.
         a. TrOCR produces text → return TrOCR result.
         b. TrOCR fails or returns nothing → fall back to the Tesseract result.

    Args:
        image_bytes: raw bytes of a PNG/JPEG/TIFF page image.

    Returns:
        OcrResult. The `backend` field records which engine produced the result
        ("tesseract" or "trocr") so callers can log / audit the decision.
    """
    # ── Stage 1: Tesseract ───────────────────────────────────────────────────
    tess_result = _tesseract_ocr_page(image_bytes)

    if tess_result.avg_confidence >= TROCR_FALLBACK_THRESHOLD:
        return tess_result

    # ── Stage 2: TrOCR fallback ──────────────────────────────────────────────
    logger.info(
        "Tesseract avg_confidence=%.3f < %.2f — handwriting suspected, "
        "escalating to TrOCR",
        tess_result.avg_confidence,
        TROCR_FALLBACK_THRESHOLD,
    )

    try:
        from python_service.ocr.trocr_backend import ocr_page as _trocr_ocr_page
        trocr_result = _trocr_ocr_page(image_bytes)

        if trocr_result.full_text.strip():
            logger.info(
                "TrOCR succeeded: %d lines, avg_conf=%.3f — using TrOCR result",
                len(trocr_result.lines),
                trocr_result.avg_confidence,
            )
            return trocr_result

        logger.warning("TrOCR returned empty text — keeping Tesseract result")

    except Exception as exc:
        logger.warning(
            "TrOCR fallback failed (%s) — keeping Tesseract result", exc
        )

    return tess_result


def get_backend_name() -> str:
    """Returns 'tesseract+trocr_fallback' to reflect the cascade."""
    return "tesseract+trocr_fallback"
