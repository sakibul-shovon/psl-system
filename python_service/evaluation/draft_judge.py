"""
Draft quality judge — independent Groq 70B evaluation.

Runs AFTER generation. Scores the draft on groundedness, completeness,
structure, and overall quality on a 1-5 scale.

Why 70B and NOT the same model as generation (Gemini)?
  A model judging its own output inflates scores — it recognises its own
  writing patterns and rates them highly regardless of actual quality.
  Groq llama-3.3-70b-versatile is completely independent of Gemini.

Why temperature=0?
  Scores must be consistent and trendable. Temperature=0 means the same
  draft always gets the same score — essential for meaningful metrics.
"""

import json
import logging
from typing import Optional

from groq import Groq

from python_service.config import settings

logger = logging.getLogger(__name__)

_client: Optional[Groq] = None

_JUDGE_PROMPT = """\
You are an independent legal document quality evaluator.

Score the following draft on each dimension using a 1-5 integer scale:
  1 = very poor  2 = poor  3 = acceptable  4 = good  5 = excellent

SCORING DIMENSIONS:
- groundedness:   Do all factual claims have inline [E1]-style citations?
                  Are cited facts actually present in the evidence provided?
- completeness:   Does the draft cover the key points from the evidence?
                  Are important clauses or obligations missing?
- structure:      Is the draft logically organized with clear sections?
                  Is it readable and professionally formatted?
- overall:        Holistic quality assessment.

EVIDENCE PROVIDED TO THE DRAFTER:
{evidence_text}

DRAFT TO EVALUATE:
{draft_text}

Return ONLY valid JSON, no explanation:
{{
  "groundedness": <1-5>,
  "completeness": <1-5>,
  "structure": <1-5>,
  "overall": <1-5>,
  "ungrounded_claims": ["list sentences that make claims without citations"],
  "missed_evidence": ["list key evidence points not reflected in the draft"],
  "reasoning": "one sentence explaining the overall score"
}}
"""


def judge_draft(
    draft_sections: list[dict],
    evidence_items: list[dict],
) -> dict:
    """
    Score a draft using Groq 70B as an independent judge.

    Args:
        draft_sections: list of section dicts from Gemini output.
        evidence_items: list of evidence dicts (each with content + evidence_id).

    Returns:
        Dict with scores and reasoning. Falls back to neutral scores on error.
    """
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY not set — skipping judge")
        return _default_scores()

    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)

    # Format evidence for the judge
    evidence_text = "\n\n".join(
        f"[{e.get('evidence_id', 'E?')}] {e.get('breadcrumb', '')}\n{e.get('content', '')[:500]}"
        for e in evidence_items
    )

    # Format draft for the judge
    draft_text = "\n\n".join(
        f"## {s.get('title', '')}\n{s.get('content', '')}"
        for s in draft_sections
    )

    prompt = _JUDGE_PROMPT.format(
        evidence_text=evidence_text[:3000],   # stay within context limits
        draft_text=draft_text[:3000],
    )

    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        scores = json.loads(response.choices[0].message.content)
        logger.info(
            "Judge scores — groundedness:%s completeness:%s structure:%s overall:%s",
            scores.get("groundedness"), scores.get("completeness"),
            scores.get("structure"), scores.get("overall"),
        )
        return scores

    except json.JSONDecodeError as exc:
        logger.error("Judge returned invalid JSON: %s", exc)
        return _default_scores()
    except Exception as exc:
        logger.error("Judge call failed: %s", exc)
        return _default_scores()


def _default_scores() -> dict:
    return {
        "groundedness": 3,
        "completeness": 3,
        "structure": 3,
        "overall": 3,
        "ungrounded_claims": [],
        "missed_evidence": [],
        "reasoning": "Judge unavailable — default scores assigned.",
    }
