"""
Legal document chunker — stack-based state machine.

Turns flat normalized lines into structured chunks that preserve
legal hierarchy (Article → Section → subsection → sub-subsection).

The stack works like nested folders:
  - Seeing "ARTICLE IV" opens a folder at depth 1
  - Seeing "Section 4.1" opens a sub-folder at depth 2
  - Seeing "Section 4.2" closes (flushes) Section 4.1 first, then opens 4.2

CRITICAL: breadcrumb is captured AT FLUSH TIME (when we pop from the stack),
not at push time. At push time we don't yet know the full ancestry.
At flush time, the stack contains the exact path from root to this section.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_TOKENS = 800        # chunks larger than this get split
OVERLAP_SENTENCES = 1   # sentences carried over between sub-chunks for context
CHARS_PER_TOKEN = 4     # rough approximation for English text

# ── Header detection patterns (ordered by level) ─────────────────────────────
_LEVEL_PATTERNS: list[tuple[int, re.Pattern]] = [
    (1, re.compile(r"^(ARTICLE|PART)\s+[IVXLCDM\d]+", re.IGNORECASE)),
    (2, re.compile(r"^(Section|SECTION)\s+\d+(\.\d+)*")),
    (3, re.compile(r"^\s*\(([a-z]|\d+)\)\s")),      # (a), (1)
    (4, re.compile(r"^\s+\([ivxlcdm]+\)\s")),         # (i), (ii)
]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LOW_CONF_RE = re.compile(r"\[LOW_CONF:([\d.]+)\]")


@dataclass
class ChunkData:
    """
    One logical section chunk — ready to be stored in SQLite + Qdrant.
    The pipeline converts this into a db.models.Chunk row.
    """
    title: str
    content: str
    breadcrumb: str
    structural_level: int
    page_range: list[int]       # [first_page, last_page]
    token_estimate: int
    ocr_confidence_avg: float
    ocr_confidence_min: float
    has_low_conf_regions: bool


@dataclass
class _StackEntry:
    """An open (not yet flushed) section sitting on the parser stack."""
    level: int
    title: str
    lines: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0


def chunk_document(pages: list[tuple[int, str]]) -> list[ChunkData]:
    """
    Main entry point. Accepts pages as (page_num, normalized_text) tuples.
    Returns ordered list of ChunkData objects.
    """
    # Flatten all pages into a single sequence of (line, page_num) pairs.
    # We process cross-page: a section that starts on page 3 and ends on page 4
    # becomes one chunk with page_range=[3,4].
    flat_lines: list[tuple[str, int]] = []
    for page_num, text in pages:
        for line in text.splitlines():
            flat_lines.append((line, page_num))

    stack: list[_StackEntry] = []
    chunks: list[ChunkData] = []
    preamble_lines: list[str] = []
    preamble_page: int = 0

    for line_text, page_num in flat_lines:
        stripped = line_text.strip()

        if not stripped:
            # Blank lines belong to whatever section is currently open
            if stack:
                stack[-1].lines.append(line_text)
                stack[-1].page_end = page_num
            else:
                preamble_lines.append(line_text)
            continue

        level = _detect_level(stripped)

        if level is not None:
            # New section header: flush all open sections at this level or deeper
            for entry, breadcrumb in _flush_from_level(stack, level):
                _create_chunks(entry, breadcrumb, chunks)

            # Open the new section
            stack.append(_StackEntry(
                level=level,
                title=stripped,
                page_start=page_num,
                page_end=page_num,
            ))

        else:
            # Regular content: goes into the innermost open section
            if stack:
                stack[-1].lines.append(line_text)
                stack[-1].page_end = page_num
            else:
                # Text before any section header = document preamble
                preamble_lines.append(line_text)
                preamble_page = page_num

    # End of document: flush everything still on the stack
    for entry, breadcrumb in _flush_from_level(stack, min_level=1):
        _create_chunks(entry, breadcrumb, chunks)

    # Preamble chunk (parties, recitals, WHEREAS clauses)
    if preamble_lines:
        preamble_text = "\n".join(preamble_lines).strip()
        if preamble_text:
            chunks.insert(0, _make_chunk(
                title="Preamble",
                content=preamble_text,
                breadcrumb="Preamble",
                level=0,
                page_start=0,
                page_end=preamble_page,
            ))

    logger.info("Chunked document: %d chunks across %d pages", len(chunks), len(pages))
    return chunks


def _detect_level(line: str) -> int | None:
    """Return structural level 1-4 if this line is a section header, else None."""
    for level, pattern in _LEVEL_PATTERNS:
        if pattern.match(line):
            return level
    return None


def _flush_from_level(
    stack: list[_StackEntry], min_level: int
) -> list[tuple[_StackEntry, str]]:
    """
    Pop all stack entries at level >= min_level, deepest first.

    The breadcrumb is built WHILE the entry is still on the stack,
    so it correctly includes the entry's own title and all its ancestors.
    Deepest-first order means child chunks appear before their parent chunks
    in the output — matching document reading order.
    """
    results = []
    while stack and stack[-1].level >= min_level:
        entry = stack[-1]
        # stack still contains this entry → breadcrumb includes its title
        breadcrumb = " > ".join(s.title for s in stack)
        stack.pop()
        results.append((entry, breadcrumb))
    return results


def _create_chunks(
    entry: _StackEntry, breadcrumb: str, output: list[ChunkData]
) -> None:
    """Convert a flushed stack entry into one or more ChunkData objects."""
    content = "\n".join(entry.lines).strip()
    if not content:
        return   # header with no body content — skip

    chunk = _make_chunk(
        title=entry.title,
        content=content,
        breadcrumb=breadcrumb,
        level=entry.level,
        page_start=entry.page_start,
        page_end=entry.page_end,
    )

    if chunk.token_estimate <= MAX_TOKENS:
        output.append(chunk)
    else:
        output.extend(_split_oversized(chunk))


def _make_chunk(
    *, title: str, content: str, breadcrumb: str,
    level: int, page_start: int, page_end: int
) -> ChunkData:
    """Build a ChunkData, computing OCR confidence from [LOW_CONF] annotations in content."""
    conf_matches = [float(c) for c in _LOW_CONF_RE.findall(content)]
    if conf_matches:
        avg_conf = sum(conf_matches) / len(conf_matches)
        min_conf = min(conf_matches)
        has_low = True
    else:
        avg_conf = 1.0
        min_conf = 1.0
        has_low = False

    return ChunkData(
        title=title,
        content=content,
        breadcrumb=breadcrumb,
        structural_level=level,
        page_range=[page_start, page_end],
        token_estimate=max(1, len(content) // CHARS_PER_TOKEN),
        ocr_confidence_avg=round(avg_conf, 3),
        ocr_confidence_min=round(min_conf, 3),
        has_low_conf_regions=has_low,
    )


def _split_oversized(chunk: ChunkData) -> list[ChunkData]:
    """
    Split a chunk that exceeds MAX_TOKENS into overlapping sub-chunks.
    Each sub-chunk gets the section title prepended so it's self-contained
    when retrieved without its neighbors.
    """
    sentences = _SENTENCE_END.split(chunk.content)
    if len(sentences) <= 1:
        return _hard_split(chunk)

    sub_chunks: list[ChunkData] = []
    current: list[str] = []
    part = 1

    for sent in sentences:
        current.append(sent)
        if len(" ".join(current)) // CHARS_PER_TOKEN >= MAX_TOKENS:
            content = chunk.title + "\n" + " ".join(current[:-1])
            sub_chunks.append(_make_chunk(
                title=f"{chunk.title} (part {part})",
                content=content,
                breadcrumb=chunk.breadcrumb,
                level=chunk.structural_level,
                page_start=chunk.page_range[0],
                page_end=chunk.page_range[1],
            ))
            part += 1
            current = current[-OVERLAP_SENTENCES:]   # carry over 1 sentence for context

    if current:
        content = chunk.title + "\n" + " ".join(current)
        sub_chunks.append(_make_chunk(
            title=f"{chunk.title} (part {part})",
            content=content,
            breadcrumb=chunk.breadcrumb,
            level=chunk.structural_level,
            page_start=chunk.page_range[0],
            page_end=chunk.page_range[1],
        ))

    logger.debug("Split '%s' (%d tokens) → %d sub-chunks", chunk.title, chunk.token_estimate, len(sub_chunks))
    return sub_chunks


def _hard_split(chunk: ChunkData) -> list[ChunkData]:
    """Fallback when no sentence boundaries found — split at character boundary."""
    max_chars = MAX_TOKENS * CHARS_PER_TOKEN
    parts = [chunk.content[i:i + max_chars] for i in range(0, len(chunk.content), max_chars)]
    return [
        _make_chunk(
            title=f"{chunk.title} (part {i + 1})",
            content=chunk.title + "\n" + part,
            breadcrumb=chunk.breadcrumb,
            level=chunk.structural_level,
            page_start=chunk.page_range[0],
            page_end=chunk.page_range[1],
        )
        for i, part in enumerate(parts)
    ]
