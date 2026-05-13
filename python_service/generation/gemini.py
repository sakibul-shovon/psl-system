"""
Gemini 2.5 Flash client — structured JSON draft generation.

Uses response_mime_type="application/json" to force valid JSON output.
The prompt (from context.py) instructs Gemini on the exact schema to follow.

Free tier: 1,500 requests/day — comfortable for development.
"""

import json
import logging

import google.generativeai as genai

from python_service.config import settings

logger = logging.getLogger(__name__)

_client_initialized = False


def _ensure_client() -> None:
    global _client_initialized
    if not _client_initialized:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        genai.configure(api_key=settings.gemini_api_key)
        _client_initialized = True


def generate_draft(prompt: str) -> dict:
    """
    Send the assembled prompt to Gemini 2.5 Flash and parse the JSON response.

    Args:
        prompt: the full prompt from context.build_prompt().

    Returns:
        Parsed dict matching the LegalDraft schema defined in context.py.
        On any failure, returns a safe error dict.
    """
    _ensure_client()

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,      # low temperature = more faithful to evidence
            max_output_tokens=8192,  # 2.5-flash uses thinking tokens; 4096 was too tight
        ),
    )

    try:
        response = model.generate_content(prompt)
        raw = response.text
        draft = json.loads(raw)
        logger.info(
            "Gemini draft generated: %d sections, overall confidence: %s",
            len(draft.get("sections", [])),
            draft.get("overallConfidence", "?"),
        )
        return draft

    except json.JSONDecodeError as exc:
        logger.error("Gemini returned invalid JSON: %s", exc)
        return _error_draft("Gemini returned malformed JSON")
    except Exception as exc:
        logger.error("Gemini generation failed: %s", exc)
        return _error_draft(str(exc))


def _error_draft(reason: str) -> dict:
    """Safe fallback when generation fails — returns a minimal valid structure."""
    return {
        "draftType": "case_fact_summary",
        "title": "Generation Failed",
        "sections": [],
        "overallConfidence": "LOW",
        "warnings": [f"Generation error: {reason}"],
    }
