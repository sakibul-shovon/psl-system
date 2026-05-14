"""
Test the OCR cascade — Tesseract → TrOCR fallback.

Usage:
    # Test with a real image file
    python -m scripts.test_ocr path/to/image.jpg

    # Test with a synthetic typed image (no real file needed)
    python -m scripts.test_ocr --typed

    # Test with a synthetic noisy image (should trigger TrOCR)
    python -m scripts.test_ocr --noisy
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _make_typed_image() -> bytes:
    """Create a clean typed-text image — Tesseract should handle this fine."""
    img = Image.new("RGB", (800, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 70), "This is a clean typed document. Tesseract should read it easily.", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_noisy_image() -> bytes:
    """Create a heavily noisy image — Tesseract will struggle (low confidence)."""
    img = Image.new("RGB", (800, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 70), "Some text", fill="black")
    arr = np.array(img)
    # Add heavy salt-and-pepper noise — makes Tesseract confidence drop
    noise_mask = np.random.random(arr.shape[:2]) < 0.4
    arr[noise_mask] = np.random.randint(0, 255, size=(noise_mask.sum(), 3))
    noisy_img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    noisy_img.save(buf, format="PNG")
    return buf.getvalue()


def test_image(image_bytes: bytes, label: str) -> None:
    from python_service.ocr.handler import ocr_page, TROCR_FALLBACK_THRESHOLD

    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Fallback threshold: Tesseract avg_confidence < {TROCR_FALLBACK_THRESHOLD}")
    print("="*60)

    result = ocr_page(image_bytes)

    print(f"\nBackend used   : {result.backend}")
    print(f"Avg confidence : {result.avg_confidence:.3f}")
    print(f"Lines found    : {len(result.lines)}")
    print(f"\nExtracted text :\n{result.full_text[:500] or '(empty)'}")

    if result.backend == "trocr":
        print("\n→ TrOCR fallback WAS triggered (Tesseract confidence was too low)")
    else:
        print(f"\n→ Tesseract result kept (confidence {result.avg_confidence:.3f} ≥ {TROCR_FALLBACK_THRESHOLD})")


def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    if "--typed" in args:
        test_image(_make_typed_image(), "Synthetic clean typed image")

    elif "--noisy" in args:
        print("NOTE: Heavy noise will trigger the TrOCR fallback.")
        print("      TrOCR model loads on first use (~30 seconds first time).")
        test_image(_make_noisy_image(), "Synthetic noisy image")

    else:
        # Real image file
        path = args[0]
        with open(path, "rb") as f:
            image_bytes = f.read()
        test_image(image_bytes, path)


if __name__ == "__main__":
    main()
