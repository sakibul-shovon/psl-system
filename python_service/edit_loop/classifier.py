"""
Edit classifier — labels what kind of change an operator made.

Uses Groq 70B to read a before/after pair and assign one of seven edit
types.  The type drives which pattern extraction logic runs next.

Edit types:
  TERMINOLOGY_CHANGE   — same meaning, different word choice
                          ("salary" → "Base Compensation")
  TONE_SHIFT           — formality level changed
                          (casual → legal-formal phrasing)
  CITATION_ADDED       — operator added a missing [E1] reference
  FACT_CORRECTION      — a factual claim was changed (dates, amounts, names)
  RESTRUCTURE          — same content, different format (prose → bullets)
  OMISSION_CORRECTION  — operator added content that was missing entirely
  NOISE                — trivial whitespace / punctuation change not worth learning

Why 70B and not 8B?
  Classification errors cascade — a NOISE label skips pattern extraction
  entirely, a wrong RESTRUCTURE label extracts the wrong rule type.
  The 3× slower 70B is worth it for label accuracy here.
"""

import json
import logging
from typing import Optional

from groq import Groq

from python_service.config import settings
from python_service.observability.langfuse_client import observe

logger = logging.getLogger(__name__)

EDIT_TYPES = [
    "TERMINOLOGY_CHANGE",
    "TONE_SHIFT",
    "CITATION_ADDED",
    "FACT_CORRECTION",
    "RESTRUCTURE",
    "OMISSION_CORRECTION",
    "NOISE",
]

_CLASSIFY_PROMPT = """\
You are an expert legal document editor classifying what type of change was made.

ORIGINAL TEXT:
{original}

EDITED TEXT:
{edited}

Classify this edit into exactly ONE of these types:
- TERMINOLOGY_CHANGE: same meaning, different word choice (e.g. "salary" → "Base Compensation")
- TONE_SHIFT: formality or register changed (e.g. casual → formal legal language)
- CITATION_ADDED: an [E1]-style evidence citation was added to a claim
- FACT_CORRECTION: a factual claim was changed (date, amount, party name, section number)
- RESTRUCTURE: same content reformatted (prose → bullets, paragraph split/merged)
- OMISSION_CORRECTION: new substantive content was added that was completely missing
- NOISE: trivial change (whitespace, punctuation only) — not worth learning from

Return ONLY valid JSON:
{{
  "edit_type": "<one of the types above>",
  "confidence": <0.0-1.0>,
  "reasoning": "one sentence explaining the classification"
}}
"""

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _client = Groq(api_key=settings.groq_api_key)
    return _client


@observe(name="groq-classify-edit")
def classify_edit(original_text: str, edited_text: str) -> dict:
    """
    Classify what kind of edit was made.

    Args:
        original_text: the draft section before operator edits.
        edited_text:   the draft section after operator edits.

    Returns:
        Dict with edit_type, confidence, reasoning.
        Falls back to NOISE classification on any failure.
    """
    prompt = _CLASSIFY_PROMPT.format(
        original=original_text[:1500],
        edited=edited_text[:1500],
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)

        edit_type = result.get("edit_type", "NOISE")
        if edit_type not in EDIT_TYPES:
            edit_type = "NOISE"

        logger.info(
            "Classified edit as %s (confidence=%.2f): %s",
            edit_type,
            result.get("confidence", 0),
            result.get("reasoning", ""),
        )
        return {
            "edit_type": edit_type,
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", ""),
        }

    except Exception as exc:
        logger.error("Edit classification failed: %s", exc)
        return {"edit_type": "NOISE", "confidence": 0.0, "reasoning": f"Error: {exc}"}
