"""
Structured field extraction via Groq 8B (JSON mode).

Extracts key legal metadata from the first 4000 characters of a document.
One LLM call per document, ~200ms. Returns a typed dict.

Why Groq 8B here (not Gemini)?
  This is a structured extraction task — identify named entities, classify
  document type. 8B is capable enough and fast. Gemini's reasoning quality
  matters for generation; here it would be wasted quota.

Why first 4000 chars?
  Legal documents front-load their key metadata: parties are identified in
  the preamble, dates and amounts appear in recitals. 4000 chars covers
  2-3 pages which is sufficient for a clean extraction.
"""

import json
import logging
from typing import Any, Optional

from groq import Groq

from python_service.config import settings

logger = logging.getLogger(__name__)

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env — cannot run structured extraction")
        _client = Groq(api_key=settings.groq_api_key)
    return _client


# The JSON schema we ask the model to fill. Spelled out in the prompt so the
# model knows exactly what fields are expected.
_EXTRACTION_PROMPT = """\
You are a legal document metadata extractor. Extract structured fields from the document text below.

Return ONLY valid JSON matching this exact schema — no markdown, no explanation:
{{
  "documentType": "string (one of: lease_agreement, service_agreement, employment_contract, nda, court_filing, settlement_agreement, corporate_resolution, unknown)",
  "parties": ["list of party names as strings"],
  "dates": ["list of important dates as strings, e.g. 'January 15, 2024'"],
  "amounts": ["list of dollar amounts as strings, e.g. '$500,000'"],
  "caseNumbers": ["list of case/reference numbers as strings, or empty list"],
  "jurisdiction": "string (state/country, or 'unknown')",
  "governingLaw": "string (governing law clause text, or 'unknown')",
  "summary": "string (one sentence describing what this document is about)"
}}

If a field cannot be determined from the text, use an empty list [] or the string "unknown".
Do not invent information not present in the text.

DOCUMENT TEXT:
{text}
"""


def extract_structured_fields(full_text: str) -> dict[str, Any]:
    """
    Extract structured metadata from a legal document.

    Args:
        full_text: the full document text (we'll truncate to 4000 chars internally).

    Returns:
        Dict matching the schema above. Always returns a complete dict;
        falls back to empty/unknown values if extraction fails.
    """
    # Truncate to first 4000 chars — covers enough for metadata extraction
    text_sample = full_text[:4000].strip()
    if not text_sample:
        logger.warning("Empty text passed to structured extractor — returning defaults")
        return _default_fields()

    prompt = _EXTRACTION_PROMPT.format(text=text_sample)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,          # deterministic for structured extraction
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        extracted = json.loads(raw)
        logger.info(
            "Structured extraction: type=%s, %d parties, %d dates",
            extracted.get("documentType", "?"),
            len(extracted.get("parties", [])),
            len(extracted.get("dates", [])),
        )
        return extracted

    except json.JSONDecodeError as exc:
        logger.error("Structured extractor returned invalid JSON: %s", exc)
        return _default_fields()
    except Exception as exc:
        logger.error("Structured extractor call failed: %s", exc)
        return _default_fields()


def _default_fields() -> dict[str, Any]:
    """Safe fallback when extraction fails — all unknowns."""
    return {
        "documentType": "unknown",
        "parties": [],
        "dates": [],
        "amounts": [],
        "caseNumbers": [],
        "jurisdiction": "unknown",
        "governingLaw": "unknown",
        "summary": "",
    }
