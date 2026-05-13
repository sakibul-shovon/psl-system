"""
OCR line normalizer.

OCR engines return physical lines — where the scanner saw line breaks
on the printed page. These don't match logical sentence or clause boundaries.

Two problems we fix here:
  1. Hyphenated word splits: "indemnifi-" + "cation" → "indemnification"
  2. Wrapped logical lines: short lines that end mid-sentence get joined
     to the next line IF the next line isn't a new structural element.

Why this matters upstream:
  - The chunker's regex patterns like `^Section \d+` must fire on a single
    line, not across a line break.
  - BM25 keyword search for "Section 4.2(b)" fails if OCR split it as
    "Sec-" / "tion 4.2(b)".
  - Embedding quality degrades when clause text is fragmented.

Input:  raw OCR text (one string with \n-separated physical lines).
Output: normalized text with logical lines.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Lines shorter than this character count are candidates for joining.
# 80 chars ≈ typical narrow-column legal document line at readable font size.
SHORT_LINE_THRESHOLD = 80

# Sentence-terminal punctuation: joining does NOT happen after these.
TERMINAL_PUNCTUATION = (".", "?", "!", ":", ";")

# Structural header patterns — a line matching any of these starts a new
# logical unit and should never be joined to the preceding line.
_STRUCTURAL_PATTERNS = [
    re.compile(r"^(ARTICLE|PART)\s+[IVXLCDM\d]+", re.IGNORECASE),
    re.compile(r"^(Section|SECTION)\s+\d+(\.\d+)*"),
    re.compile(r"^\s*\(([a-z]|\d+)\)\s"),        # subsection: (a), (1)
    re.compile(r"^\s+\([ivxlcdm]+\)\s"),           # sub-subsection: (i), (ii)
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),             # all-caps heading (at least 5 chars)
    re.compile(r"^\d+\.\s+[A-Z]"),                 # numbered clause: "1. Party"
    re.compile(r"^WHEREAS"),
    re.compile(r"^NOW,\s+THEREFORE"),
    re.compile(r"^IN WITNESS"),
    re.compile(r"^\[LOW_CONF:"),                    # keep low-conf annotation lines intact
]

# Matches a trailing hyphen that indicates a mid-word line break.
# The hyphen must be at the very end (possibly before \n).
_TRAILING_HYPHEN = re.compile(r"-\s*$")


def normalize_lines(raw_text: str) -> str:
    """
    Normalize OCR physical lines into logical lines.

    Args:
        raw_text: output from ocr_page() — newline-separated physical lines,
                  possibly with [LOW_CONF:0.54] prefix annotations.

    Returns:
        Normalized text where logical sentences are on single lines.
    """
    if not raw_text.strip():
        return raw_text

    lines = raw_text.splitlines()
    result: list[str] = []
    i = 0

    while i < len(lines):
        current = lines[i]

        # ── Pass 1: hyphen join ──────────────────────────────────────────
        # "indemnifi-" + "cation" → "indemnification"
        # Only join if current ends with a hyphen AND next line starts
        # with a lowercase letter (continuation of the same word).
        if (
            i + 1 < len(lines)
            and _TRAILING_HYPHEN.search(current)
            and lines[i + 1]
            and lines[i + 1][0].islower()
        ):
            # Strip the trailing hyphen, glue directly to next word fragment
            base = _TRAILING_HYPHEN.sub("", current).rstrip()
            continuation = lines[i + 1].lstrip()
            current = base + continuation
            i += 2   # consumed two lines
            # Don't append yet — this merged line might itself be short
            # and eligible for further joining. Re-insert at the same position.
            lines.insert(i, current)
            continue

        # ── Pass 2: logical line join ────────────────────────────────────
        # Join to next line when ALL three conditions hold:
        #   a) current line doesn't end with sentence-terminal punctuation
        #   b) current line is "short" (likely a wrapped visual line)
        #   c) next line isn't a structural header
        if (
            i + 1 < len(lines)
            and not current.rstrip().endswith(TERMINAL_PUNCTUATION)
            and len(current.rstrip()) < SHORT_LINE_THRESHOLD
            and not _is_structural(lines[i + 1])
            and lines[i + 1].strip()   # don't join into a blank line
        ):
            # Join with a space, then move to the next line and repeat
            # the check (the joined line might itself be joinable)
            current = current.rstrip() + " " + lines[i + 1].lstrip()
            i += 2
            # Re-insert merged line to be processed again
            lines.insert(i, current)
            continue

        result.append(current)
        i += 1

    normalized = "\n".join(result)
    original_line_count = len(raw_text.splitlines())
    logger.debug(
        "Line normalization: %d physical lines → %d logical lines",
        original_line_count, len(result),
    )
    return normalized


def _is_structural(line: str) -> bool:
    """Return True if the line looks like a new section/article header."""
    stripped = line.strip()
    return any(pat.match(stripped) for pat in _STRUCTURAL_PATTERNS)


def normalize_document(page_texts: list[str]) -> list[str]:
    """
    Normalize all pages of a document.

    Args:
        page_texts: list of raw text strings, one per page.

    Returns:
        List of normalized text strings (same length, same order).
    """
    return [normalize_lines(text) for text in page_texts]
