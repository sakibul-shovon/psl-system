"""
Unit tests for the legal chunker.

Tests the stack-based state machine with synthetic legal text — no file I/O,
no LLM calls, no Qdrant.  Fast and deterministic.

Run with:  python -m pytest tests/test_chunker.py -v
"""

import pytest
from python_service.chunking.legal_chunker import chunk_document, ChunkData

# ── Fixtures ──────────────────────────────────────────────────────────────────

SIMPLE_CONTRACT = """
ARTICLE I DEFINITIONS

Section 1.1 Base Compensation
As used in this Agreement, "Base Compensation" means the annual salary payable
to Employee as established from time to time by the Board of Directors.
The Base Compensation shall be reviewed annually.

Section 1.2 Effective Date
The Effective Date of this Agreement shall be January 1, 2024.
All provisions herein shall take effect on the Effective Date.

ARTICLE II TERMINATION

Section 2.1 Termination by Company
The Company may terminate Employee's employment for any reason at any time
upon thirty (30) days written notice delivered to Employee.

Section 2.2 Termination by Employee
Employee may terminate this Agreement by providing thirty (30) days
written notice to the Company.
"""

OVERSIZED_TEXT = " ".join(["This is a very long sentence about legal matters."] * 100)

OVERSIZED_CONTRACT = f"""
ARTICLE I COMPENSATION

Section 1.1 Payment Terms
{OVERSIZED_TEXT}
"""


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_basic_chunking_returns_chunks():
    """chunker produces at least one chunk from valid legal text."""
    chunks = chunk_document([(1, SIMPLE_CONTRACT)])
    assert len(chunks) > 0


def test_chunk_type():
    """every returned item is a ChunkData instance."""
    chunks = chunk_document([(1, SIMPLE_CONTRACT)])
    for c in chunks:
        assert isinstance(c, ChunkData)


def test_breadcrumb_includes_article_and_section():
    """breadcrumbs capture the hierarchy: Article > Section."""
    chunks = chunk_document([(1, SIMPLE_CONTRACT)])
    breadcrumbs = [c.breadcrumb for c in chunks]
    # At least one chunk should have a multi-level breadcrumb
    multi_level = [b for b in breadcrumbs if ">" in b]
    assert len(multi_level) > 0, f"No multi-level breadcrumbs found. Got: {breadcrumbs}"


def test_article_titles_captured():
    """chunks from Article I should reference it in title or breadcrumb."""
    chunks = chunk_document([(1, SIMPLE_CONTRACT)])
    article_chunks = [
        c for c in chunks
        if "ARTICLE I" in c.breadcrumb or "ARTICLE I" in c.title
        or "DEFINITIONS" in c.breadcrumb or "DEFINITIONS" in c.title
    ]
    assert len(article_chunks) > 0


def test_content_non_empty():
    """every chunk has non-empty content."""
    chunks = chunk_document([(1, SIMPLE_CONTRACT)])
    for c in chunks:
        assert c.content.strip(), f"Empty content in chunk: {c.title}"


def test_token_estimate_positive():
    """token_estimate must be > 0 for every chunk."""
    chunks = chunk_document([(1, SIMPLE_CONTRACT)])
    for c in chunks:
        assert c.token_estimate > 0


def test_structural_level_in_range():
    """structural_level must be 1–4."""
    chunks = chunk_document([(1, SIMPLE_CONTRACT)])
    for c in chunks:
        assert 1 <= c.structural_level <= 4, (
            f"structural_level={c.structural_level} out of range for chunk '{c.title}'"
        )


def test_oversized_chunk_is_split():
    """a section exceeding MAX_TOKENS gets split into multiple sub-chunks."""
    chunks = chunk_document([(1, OVERSIZED_CONTRACT)])
    # The oversized section should produce more than 1 chunk
    section_chunks = [c for c in chunks if "Payment Terms" in c.title or "1.1" in c.title]
    assert len(section_chunks) > 1, (
        f"Expected oversized section to be split, got {len(section_chunks)} chunk(s)"
    )


def test_default_ocr_confidence():
    """chunks from plain text (no OCR) have full confidence."""
    chunks = chunk_document([(1, SIMPLE_CONTRACT)])
    for c in chunks:
        assert c.ocr_confidence_avg == 1.0
        assert not c.has_low_conf_regions


def test_multiple_articles_produce_multiple_chunks():
    """two articles in the text → chunks from both."""
    chunks = chunk_document([(1, SIMPLE_CONTRACT)])
    titles_and_breadcrumbs = " ".join(c.breadcrumb + c.title for c in chunks)
    assert "ARTICLE I" in titles_and_breadcrumbs or "DEFINITIONS" in titles_and_breadcrumbs
    assert "ARTICLE II" in titles_and_breadcrumbs or "TERMINATION" in titles_and_breadcrumbs
