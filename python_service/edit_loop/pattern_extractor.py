"""
Pattern extractor — turns a classified edit into a reusable style rule.

Given a before/after pair and its edit_type label, asks Groq 70B to
generalize the specific change into an imperative rule that can be applied
to future documents.

The extracted rule is intentionally generalized — not just "change salary
to Base Compensation in this contract" but "always use 'Base Compensation'
when referring to the employee's fixed pay in employment agreements."

Why generalize?
  A specific example is useless for the next document.  The generalized rule
  is what gets injected into the next Gemini prompt via context.py, so it
  must be actionable for any document of the same type.

NOISE edits return None — nothing is learned from them.
"""

import json
import logging
from typing import Optional

from groq import Groq

from python_service.config import settings

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
You are a legal writing expert extracting a reusable style rule from an editor's correction.

EDIT TYPE: {edit_type}

ORIGINAL (before edit):
{original}

EDITED (after edit):
{edited}

DOCUMENT TYPE: {document_type}
DRAFT TYPE: {draft_type}

Extract a generalized, reusable rule from this correction.
The rule must be:
1. Written as an imperative instruction (start with a verb: "Use", "Always", "Prefer", "Avoid")
2. General enough to apply to other documents of the same type
3. Accompanied by a short before/after example that illustrates the rule

Return ONLY valid JSON:
{{
  "rule_type": "{edit_type_lower}",
  "description": "imperative one-sentence rule",
  "few_shot_before": "short example of the WRONG way (≤30 words)",
  "few_shot_after": "short example of the CORRECT way (≤30 words)",
  "confidence": <0.0-1.0>,
  "applicable_document_types": ["{document_type}"],
  "applicable_draft_types": ["{draft_type}"],
  "applicable_section_types": ["compensation", "termination", "general"]
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


def extract_pattern(
    original_text: str,
    edited_text: str,
    edit_type: str,
    document_type: str = "unknown",
    draft_type: str = "case_fact_summary",
) -> Optional[dict]:
    """
    Extract a reusable rule from a classified edit.

    Args:
        original_text: before-edit text.
        edited_text:   after-edit text.
        edit_type:     classification label from classifier.py.
        document_type: e.g. "employment_contract".
        draft_type:    e.g. "case_fact_summary".

    Returns:
        Pattern dict with rule_type, description, few_shot_before/after,
        confidence, applicable_* lists.
        Returns None if the edit_type is NOISE (nothing to learn).
    """
    if edit_type == "NOISE":
        return None

    prompt = _EXTRACT_PROMPT.format(
        edit_type=edit_type,
        edit_type_lower=edit_type.lower(),
        original=original_text[:1200],
        edited=edited_text[:1200],
        document_type=document_type,
        draft_type=draft_type,
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)

        logger.info(
            "Extracted pattern [%s]: %s (conf=%.2f)",
            result.get("rule_type", "?"),
            result.get("description", "")[:80],
            result.get("confidence", 0),
        )
        return result

    except Exception as exc:
        logger.error("Pattern extraction failed: %s", exc)
        return None
