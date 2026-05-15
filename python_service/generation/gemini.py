"""
Draft generation — uses Groq (Llama 3.3 70B) with JSON mode.

Switched from Gemini 2.5 Flash because free-tier Gemini quota is exhausted
during testing. Groq has a more generous free tier and the key is already
configured for judging and classification.

JSON mode is enforced via response_format={"type": "json_object"}.
"""

import json
import logging
import time

from groq import Groq

from python_service.config import settings
from python_service.observability.langfuse_client import observe

logger = logging.getLogger(__name__)

_client: Groq | None = None
_MODEL = "llama-3.3-70b-versatile"
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 10


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        _client = Groq(api_key=settings.groq_api_key)
    return _client


@observe(name="groq-generate-draft")
def generate_draft(prompt: str) -> dict:
    """
    Send the assembled prompt to Groq Llama 3.3 70B and parse the JSON response.
    Retries with exponential backoff on rate limit errors.
    """
    client = _get_client()

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a legal drafting assistant. "
                            "Always respond with valid JSON matching the schema in the user prompt. "
                            "Never include markdown code fences or any text outside the JSON object."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=8192,
            )
            raw = response.choices[0].message.content
            draft = json.loads(raw)
            logger.info(
                "Groq draft generated: %d sections, overall confidence: %s",
                len(draft.get("sections", [])),
                draft.get("overallConfidence", "?"),
            )
            return draft

        except json.JSONDecodeError as exc:
            logger.error("Groq returned invalid JSON: %s", exc)
            return _error_draft("Groq returned malformed JSON")

        except Exception as exc:
            err_str = str(exc)
            if ("429" in err_str or "rate" in err_str.lower()) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Groq rate limit (attempt %d/%d) — waiting %ds",
                    attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                last_exc = exc
                continue
            logger.error("Groq generation failed: %s", exc)
            return _error_draft(err_str)

    return _error_draft(f"Groq rate-limited after {_MAX_RETRIES} retries: {last_exc}")


def _error_draft(reason: str) -> dict:
    """Safe fallback when generation fails."""
    return {
        "draftType": "case_fact_summary",
        "title": "Generation Failed",
        "sections": [],
        "overallConfidence": "LOW",
        "warnings": [f"Generation error: {reason}"],
    }
