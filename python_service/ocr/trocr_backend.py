"""
TrOCR fallback backend — Microsoft's handwriting-specialist transformer.

Used ONLY when Tesseract's avg_confidence < 0.35 (the signal that the page
is likely handwritten or heavily degraded and Tesseract cannot read it).

Why TrOCR?
  Tesseract is trained on printed fonts. Its character recogniser breaks down
  on cursive or mixed-case handwriting. TrOCR (trocr-base-handwritten) is a
  ViT + GPT-2 model fine-tuned on the IAM and CVL handwriting datasets — it is
  specifically designed for the case Tesseract fails.

Why not always use TrOCR?
  Speed: ~2–5 s per text line on CPU. A 20-line page takes ~60–100 s.
  For typed documents (Tesseract ≥ 0.35) there is no quality gain — only cost.
  TrOCR is always the fallback, never the first choice.

Performance note:
  Model is lazy-loaded on first use. One-time download ~400 MB (already have
  torch + transformers from sentence-transformers). GPU optional — falls back
  to CPU automatically.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from python_service.ocr.tesseract_backend import LOW_CONF_THRESHOLD, OcrLine, OcrResult

logger = logging.getLogger(__name__)

# ── Model identifiers ──────────────────────────────────────────────────────────
TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"

# TrOCR does not produce per-token confidence like Tesseract. We assign a fixed
# moderate score so downstream code (chunker, evidence packager) still sees this
# page as uncertain — it WAS hard enough to fail Tesseract.
TROCR_CONF_SCORE = 0.60

# ── Lazy singletons — loaded once on first handwriting page ───────────────────
_processor = None
_model     = None


def _ensure_model() -> None:
    """Load TrOCR processor and model on first call. No-op on subsequent calls."""
    global _processor, _model
    if _processor is not None:
        return

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        raise RuntimeError(
            "transformers is required for TrOCR but not installed. "
            "It should already be present via sentence-transformers. "
            "Try: pip install transformers"
        )

    logger.info(
        "TrOCR: loading %s for first time (~400 MB, CPU — this takes ~30 s)...",
        TROCR_MODEL_ID,
    )
    _processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
    _model     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
    _model.eval()
    logger.info("TrOCR model ready")


def ocr_page(image_bytes: bytes) -> OcrResult:
    """
    Run TrOCR on a single page image.

    Steps:
      1. Decode bytes → OpenCV image
      2. Binarize (Otsu) to find text line bounding boxes
      3. Crop each text line from the ORIGINAL colour image
      4. Run TrOCR on each cropped line (RGB PIL image)
      5. Assemble OcrResult with fixed confidence score

    Args:
        image_bytes: raw PNG/JPEG/TIFF bytes of one page.

    Returns:
        OcrResult with backend="trocr".
        avg_confidence is fixed at TROCR_CONF_SCORE (0.60) — better than
        Tesseract's failure-trigger threshold but honestly flagged as uncertain.
    """
    _ensure_model()

    # ── 1. Decode ────────────────────────────────────────────────────────────
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("TrOCR: could not decode image bytes")

    # ── 2. Binarize for line detection (NOT passed to TrOCR) ─────────────────
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    boxes  = _detect_line_boxes(bw)

    if not boxes:
        logger.warning("TrOCR: no text line regions detected — blank or unreadable page")
        return OcrResult(lines=[], full_text="", avg_confidence=0.0, backend="trocr")

    logger.info("TrOCR: recognised %d line region(s), running inference...", len(boxes))

    # ── 3 + 4. Crop lines from colour image → TrOCR inference ────────────────
    lines: list[OcrLine] = []
    for x, y, w, h in boxes:
        crop_bgr = img_bgr[y : y + h, x : x + w]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_line = Image.fromarray(crop_rgb)

        try:
            pixel_values  = _processor(pil_line, return_tensors="pt").pixel_values
            generated_ids = _model.generate(pixel_values)
            text          = _processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
        except Exception as exc:
            logger.debug("TrOCR inference error on line crop: %s", exc)
            continue

        if text:
            is_low = TROCR_CONF_SCORE < LOW_CONF_THRESHOLD
            lines.append(
                OcrLine(text=text, confidence=TROCR_CONF_SCORE, is_low_conf=is_low)
            )

    if not lines:
        logger.warning("TrOCR: inference returned no text for any line")
        return OcrResult(lines=[], full_text="", avg_confidence=0.0, backend="trocr")

    # ── 5. Assemble full_text ─────────────────────────────────────────────────
    assembled = []
    for line in lines:
        if line.is_low_conf:
            assembled.append(f"[LOW_CONF:{line.confidence:.2f}] {line.text}")
        else:
            assembled.append(line.text)

    full_text  = "\n".join(assembled)
    avg_conf   = round(sum(l.confidence for l in lines) / len(lines), 3)

    logger.info("TrOCR: %d lines, avg_conf=%.3f", len(lines), avg_conf)
    return OcrResult(lines=lines, full_text=full_text, avg_confidence=avg_conf, backend="trocr")


def _detect_line_boxes(binary_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Find text-line bounding boxes using horizontal morphological dilation.

    Approach:
      - Invert so text pixels are white (morphological ops work on white regions)
      - Dilate horizontally: this merges all words on the same line into one blob
      - findContours: each blob → one text line
      - Filter by minimum width (skip noise) and add vertical padding

    Returns:
        [(x, y, w, h)] sorted top-to-bottom (reading order).
    """
    h_img, w_img = binary_img.shape

    # White text on black for morphological operations
    inverted = cv2.bitwise_not(binary_img)

    # Wide horizontal kernel merges words → single-line blobs.
    # Kernel width (80px) larger than typical inter-word gap; height (8px) covers
    # normal line heights without merging adjacent lines.
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 8))
    dilated = cv2.dilate(inverted, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Reject noise: a real text line is at least 10% of page width and ≥12px tall
        if w < w_img * 0.10 or h < 12:
            continue

        # Add small vertical padding so TrOCR sees ascenders/descenders fully
        pad = 6
        boxes.append((
            max(0,       x),
            max(0,       y - pad),
            min(w_img,   x + w) - x,
            min(h_img,   y + h + pad) - max(0, y - pad),
        ))

    return sorted(boxes, key=lambda b: b[1])   # top-to-bottom reading order
