"""
File router — per-page routing between text layer and OCR.

PDFs are mixed: some pages have a proper text layer (direct copy-paste
from a word processor), others are scanned images embedded in a PDF wrapper.
We inspect each page individually rather than treating the whole PDF one way.

Decision rule:
  - Extract text via pypdf.
  - If meaningful text length >= MIN_TEXT_CHARS → TEXT_LAYER (use it directly).
  - Otherwise → OCR_NEEDED (render the page to an image, run OCR).

For image files (JPG, PNG, TIFF): always OCR_NEEDED — there is no text layer.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from pypdf import PdfReader

from python_service.ocr.handler import OcrResult, ocr_page

logger = logging.getLogger(__name__)

# If a page has fewer than this many non-whitespace characters after pypdf
# extraction, we treat it as a scan and send it through OCR.
MIN_TEXT_CHARS = 20

# Image formats we accept directly (no PDF wrapper needed).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

# PDF page render resolution. 150 DPI is the minimum for readable OCR;
# 200 DPI is better for degraded scans but uses more memory.
RENDER_DPI = 150


@dataclass
class PageContent:
    """
    All information about one page after routing.

    Either raw_text comes from the PDF text layer (routing=TEXT_LAYER)
    or from OCR (routing=OCR_NEEDED). Either way, the caller gets a
    uniform object and doesn't need to know which path ran.
    """
    page_num: int                   # 0-indexed page number
    routing: str                    # "TEXT_LAYER" | "OCR_NEEDED"
    raw_text: str                   # text content (with [LOW_CONF] annotations if OCR)
    ocr_result: Optional[OcrResult] = None   # populated only when routing=OCR_NEEDED
    image_bytes: Optional[bytes] = None      # PNG bytes of rendered page (kept for debugging)


@dataclass
class RoutedDocument:
    """
    Full routing result for one uploaded file.
    """
    file_path: str
    file_type: str                  # "pdf" | "image"
    pages: list[PageContent] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenate all pages' text for document-level operations (e.g. field extraction)."""
        return "\n\n".join(p.raw_text for p in self.pages if p.raw_text.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for p in self.pages if p.routing == "OCR_NEEDED")


def route_file(file_path: Path) -> RoutedDocument:
    """
    Route a PDF or image file, producing a RoutedDocument.

    Args:
        file_path: path to the uploaded file.

    Returns:
        RoutedDocument with one PageContent per page.

    Raises:
        ValueError: if the file extension is not supported.
        FileNotFoundError: if the file doesn't exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _route_pdf(file_path)
    elif suffix in IMAGE_EXTENSIONS:
        return _route_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Accepted: pdf, jpg, png, tiff")


def _route_pdf(file_path: Path) -> RoutedDocument:
    """
    Per-page PDF routing.

    Opens the PDF twice:
    - Once with pypdf (fast text extraction)
    - Once with pypdfium2 (high-quality page-to-image rendering, only for OCR pages)
    """
    result = RoutedDocument(file_path=str(file_path), file_type="pdf")

    # pypdf for text extraction
    reader = PdfReader(str(file_path))

    # pypdfium2 for image rendering — opened lazily (only used if any page needs OCR)
    pdf_doc = pdfium.PdfDocument(str(file_path))

    for page_num, pdf_page in enumerate(reader.pages):
        extracted = pdf_page.extract_text() or ""
        meaningful_chars = len(extracted.replace(" ", "").replace("\n", ""))

        if meaningful_chars >= MIN_TEXT_CHARS:
            # Good text layer — use it directly.
            logger.debug("Page %d: TEXT_LAYER (%d chars)", page_num, meaningful_chars)
            result.pages.append(PageContent(
                page_num=page_num,
                routing="TEXT_LAYER",
                raw_text=extracted.strip(),
            ))
        else:
            # Thin or empty text — render to image and OCR.
            logger.debug("Page %d: OCR_NEEDED (only %d chars extracted)", page_num, meaningful_chars)
            img_bytes = _render_pdf_page(pdf_doc, page_num)
            ocr_result = ocr_page(img_bytes)
            result.pages.append(PageContent(
                page_num=page_num,
                routing="OCR_NEEDED",
                raw_text=ocr_result.full_text,
                ocr_result=ocr_result,
                image_bytes=img_bytes,
            ))

    pdf_doc.close()

    ocr_count = result.ocr_page_count
    logger.info(
        "Routed '%s': %d pages total, %d OCR, %d text-layer",
        file_path.name, result.page_count, ocr_count, result.page_count - ocr_count,
    )
    return result


def _route_image(file_path: Path) -> RoutedDocument:
    """
    Image files are always OCR — there's no text layer to extract from.
    """
    result = RoutedDocument(file_path=str(file_path), file_type="image")
    img_bytes = file_path.read_bytes()
    ocr_result = ocr_page(img_bytes)
    result.pages.append(PageContent(
        page_num=0,
        routing="OCR_NEEDED",
        raw_text=ocr_result.full_text,
        ocr_result=ocr_result,
        image_bytes=img_bytes,
    ))
    logger.info("Routed image '%s': 1 page, OCR", file_path.name)
    return result


def _render_pdf_page(pdf_doc: pdfium.PdfDocument, page_num: int) -> bytes:
    """
    Render one PDF page to PNG bytes at RENDER_DPI.

    pypdfium2 renders using the system's PDFium library (bundled in the wheel)
    so it doesn't need poppler or ghostscript installed.
    """
    page = pdf_doc[page_num]
    scale = RENDER_DPI / 72.0   # PDF points are 72 DPI; scale to target resolution
    bitmap = page.render(scale=scale, rotation=0)
    pil_image = bitmap.to_pil()

    from io import BytesIO
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()
