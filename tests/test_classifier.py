"""
Unit tests for the edit classifier.

Mocks the Groq API so tests run instantly with no network calls.
Verifies that the classifier correctly maps edit types and handles errors.

Run with:  python -m pytest tests/test_classifier.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from python_service.edit_loop.classifier import classify_edit, EDIT_TYPES

# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_groq_response(edit_type: str, confidence: float = 0.9, reasoning: str = "test"):
    """Build a fake Groq API response object."""
    content = json.dumps({
        "edit_type": edit_type,
        "confidence": confidence,
        "reasoning": reasoning,
    })
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("python_service.edit_loop.classifier._get_client")
def test_terminology_change_classified(mock_get_client):
    """salary → Base Compensation should be TERMINOLOGY_CHANGE."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        "TERMINOLOGY_CHANGE", confidence=0.95,
        reasoning="Same meaning, different legal term used."
    )
    mock_get_client.return_value = mock_client

    result = classify_edit(
        original_text="Employee will receive a salary of $150,000 per year.",
        edited_text="Employee shall receive Base Compensation of $150,000 per annum.",
    )

    assert result["edit_type"] == "TERMINOLOGY_CHANGE"
    assert result["confidence"] == 0.95
    assert "reasoning" in result


@patch("python_service.edit_loop.classifier._get_client")
def test_noise_classified(mock_get_client):
    """Whitespace-only change should be NOISE."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        "NOISE", confidence=0.99, reasoning="Only whitespace changed."
    )
    mock_get_client.return_value = mock_client

    result = classify_edit(
        original_text="Employee shall receive Base Compensation.",
        edited_text="Employee shall receive Base Compensation.  ",
    )

    assert result["edit_type"] == "NOISE"


@patch("python_service.edit_loop.classifier._get_client")
def test_citation_added_classified(mock_get_client):
    """Adding [E1] citation should be CITATION_ADDED."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        "CITATION_ADDED", confidence=0.98, reasoning="Evidence citation was added."
    )
    mock_get_client.return_value = mock_client

    result = classify_edit(
        original_text="The Company shall pay a lump sum within 15 days.",
        edited_text="The Company shall pay a lump sum within 15 days [E1].",
    )

    assert result["edit_type"] == "CITATION_ADDED"


@patch("python_service.edit_loop.classifier._get_client")
def test_unknown_type_falls_back_to_noise(mock_get_client):
    """An unrecognised edit_type from the LLM should be forced to NOISE."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        "INVALID_TYPE_XYZ", confidence=0.5
    )
    mock_get_client.return_value = mock_client

    result = classify_edit("before text", "after text")
    assert result["edit_type"] == "NOISE"


@patch("python_service.edit_loop.classifier._get_client")
def test_api_failure_returns_noise(mock_get_client):
    """If the Groq API raises an exception, return NOISE with confidence 0."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API timeout")
    mock_get_client.return_value = mock_client

    result = classify_edit("before", "after")
    assert result["edit_type"] == "NOISE"
    assert result["confidence"] == 0.0


def test_all_edit_types_are_strings():
    """EDIT_TYPES constant should be a list of non-empty strings."""
    assert isinstance(EDIT_TYPES, list)
    assert len(EDIT_TYPES) > 0
    for t in EDIT_TYPES:
        assert isinstance(t, str) and t.strip()


@patch("python_service.edit_loop.classifier._get_client")
def test_tone_shift_classified(mock_get_client):
    """Casual → formal rewrite should be TONE_SHIFT."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        "TONE_SHIFT", confidence=0.88, reasoning="Register changed from casual to formal legal."
    )
    mock_get_client.return_value = mock_client

    result = classify_edit(
        original_text="The employee will get 3x their salary if fired.",
        edited_text="Employee shall receive a lump sum equal to three (3) times Base Compensation upon termination.",
    )

    assert result["edit_type"] == "TONE_SHIFT"
    assert result["confidence"] > 0
