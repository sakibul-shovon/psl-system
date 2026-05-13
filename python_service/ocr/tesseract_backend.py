"""
Tesseract OCR backend.

Takes raw image bytes, preprocesses with OpenCV, runs pytesseract,
and returns a structured result with per-line confidence scores.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image

from python_service.config import settings

logger = logging.getLogger(__name__)

# Tell pytesseract where the binary lives (from config, e.g. C:\Program Files\...)
pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


@dataclass
class OcrLine:
    text: str
    confidence: float   # 0.0–1.0
    is_low_conf: bool   # True when confidence < LOW_CONF_THRESHOLD


@dataclass
class OcrResult:
    lines: list[OcrLine]
    full_text: str
    avg_confidence: float
    backend: str = "tesseract"


# Lines below this threshold get flagged and annotated in the assembled text.
LOW_CONF_THRESHOLD = 0.70


def ocr_page(image_bytes: bytes) -> OcrResult:
    """
    Run Tesseract on a single page image.

    Steps:
      1. Decode bytes → OpenCV image
      2. Preprocess: grayscale → Otsu binarize → deskew → denoise
      3. Run pytesseract in data mode (returns per-word confidence)
      4. Aggregate word confidence → line confidence
      5. Return OcrResult with structured line data
    """
    # ── 1. Decode ───────────────────────────────────────────────────────────
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes — unsupported format?")

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu binarization: automatically picks the threshold between text and background.
    # Works well on scanned documents where background isn't pure white.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew: measure skew angle via Hough line transform and rotate to correct it.
    # Skewed text degrades Tesseract accuracy significantly.
    binary = _deskew(binary)

    # Light denoising — removes salt-and-pepper noise from fax/scan artifacts.
    # Kernel (3,3) is small enough not to blur character edges.
    binary = cv2.medianBlur(binary, 3)

    # ── 3. Tesseract data mode ───────────────────────────────────────────────
    # image_to_data returns a TSV-formatted string with one row per word.
    # Each row contains: level, page_num, block_num, par_num, line_num, word_num,
    # left, top, width, height, conf, text
    pil_img = Image.fromarray(binary)
    tsv_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

    # ── 4. Aggregate per-word data into lines ────────────────────────────────
    # Group words by their line_num (within the same block+paragraph).
    # Tesseract reports confidence as 0–100 (-1 means non-text row).
    lines: list[OcrLine] = []
    line_key_to_words: dict[tuple, list] = {}

    for i, conf in enumerate(tsv_data["conf"]):
        if conf == -1:          # non-word rows (page, block, paragraph headers)
            continue
        word = tsv_data["text"][i].strip()
        if not word:
            continue

        # Group key = (block, paragraph, line) — unique per visual text line
        key = (tsv_data["block_num"][i], tsv_data["par_num"][i], tsv_data["line_num"][i])
        if key not in line_key_to_words:
            line_key_to_words[key] = []
        line_key_to_words[key].append({"text": word, "conf": int(conf)})

    for key in sorted(line_key_to_words):
        words = line_key_to_words[key]
        line_text = " ".join(w["text"] for w in words)
        # Convert confidence from 0-100 to 0.0-1.0 and average across words
        avg_conf = sum(w["conf"] for w in words) / (len(words) * 100)
        is_low = avg_conf < LOW_CONF_THRESHOLD
        lines.append(OcrLine(text=line_text, confidence=round(avg_conf, 3), is_low_conf=is_low))

    if not lines:
        logger.warning("Tesseract returned no text — image may be blank or corrupt")
        return OcrResult(lines=[], full_text="", avg_confidence=0.0)

    # ── 5. Assemble full_text with low-confidence annotations ───────────────
    # Low-confidence lines are tagged so the line normalizer and chunker can
    # propagate this signal through to the draft.
    assembled: list[str] = []
    for line in lines:
        if line.is_low_conf:
            assembled.append(f"[LOW_CONF:{line.confidence:.2f}] {line.text}")
        else:
            assembled.append(line.text)

    full_text = "\n".join(assembled)
    total_avg = round(sum(ln.confidence for ln in lines) / len(lines), 3)

    logger.debug("Tesseract: %d lines, avg_conf=%.3f", len(lines), total_avg)
    return OcrResult(lines=lines, full_text=full_text, avg_confidence=total_avg)


def _deskew(binary_img: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew angle using Hough line transform.
    Returns the rotated image. If skew detection fails, returns the original.
    """
    try:
        # Find edges, then detect lines. HoughLinesP returns line segments.
        edges = cv2.Canny(binary_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return binary_img

        # Measure angle of each detected line segment
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines (text baselines)
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return binary_img

        median_angle = np.median(angles)
        # Don't rotate if essentially straight — avoids degrading already clean scans
        if abs(median_angle) < 0.5:
            return binary_img

        h, w = binary_img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(binary_img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        logger.debug("Deskewed by %.2f degrees", median_angle)
        return rotated
    except Exception as exc:
        logger.warning("Deskew failed (%s), continuing without rotation", exc)
        return binary_img
