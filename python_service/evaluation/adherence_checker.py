"""
Adherence checker — verifies Gemini followed the injected style patterns.

After generation, scans the draft text for the few_shot_before phrase from
each injected pattern.  If the "wrong" phrasing still appears in the draft,
the model ignored the rule.

Why substring search instead of NLI?
  Pattern rules are specific word-choice rules ("use X not Y").  The
  few_shot_before is the exact wrong phrase to avoid.  Substring search
  is fast, deterministic, and correct for this narrow task.  NLI would
  add 200ms of model inference for a problem that string matching solves
  perfectly.

Adherence score = patterns_followed / patterns_injected
  1.0 = Gemini followed all rules
  0.0 = Gemini ignored all rules
"""

import logging

logger = logging.getLogger(__name__)


def check_adherence(
    draft_sections: list[dict],
    injected_patterns: list[dict],
) -> dict:
    """
    Check whether the draft followed each injected pattern.

    Args:
        draft_sections:    list of section dicts from Gemini output.
        injected_patterns: patterns that were injected into the prompt
                           (each must have fewShotBefore and description).

    Returns:
        Dict with adherence_score (0.0–1.0), followed/violated lists,
        and per-pattern detail.
    """
    if not injected_patterns:
        return {
            "adherence_score": 1.0,
            "patterns_injected": 0,
            "patterns_followed": 0,
            "patterns_violated": 0,
            "violations": [],
            "detail": [],
        }

    # Flatten draft to one searchable string (lowercased for case-insensitive match)
    full_text = " ".join(
        s.get("content", "") for s in draft_sections
    ).lower()

    followed = 0
    violated = 0
    violations: list[str] = []
    detail: list[dict] = []

    for pattern in injected_patterns:
        before_phrase = pattern.get("fewShotBefore", "").strip().lower()
        description = pattern.get("description", "")

        if not before_phrase:
            # Can't check adherence without a before-phrase — count as followed
            followed += 1
            detail.append({
                "pattern_id": pattern.get("pattern_id"),
                "description": description,
                "result": "UNCHECKED",
                "reason": "no few_shot_before phrase to search for",
            })
            continue

        if before_phrase in full_text:
            # Wrong phrase still present → model ignored the rule
            violated += 1
            violations.append(description)
            detail.append({
                "pattern_id": pattern.get("pattern_id"),
                "description": description,
                "result": "VIOLATED",
                "found_phrase": pattern.get("fewShotBefore"),
            })
            logger.info("Pattern VIOLATED — found %r in draft", before_phrase[:50])
        else:
            followed += 1
            detail.append({
                "pattern_id": pattern.get("pattern_id"),
                "description": description,
                "result": "FOLLOWED",
            })

    total = len(injected_patterns)
    score = round(followed / total, 3) if total > 0 else 1.0

    logger.info(
        "Adherence: %d/%d patterns followed → score=%.2f",
        followed, total, score,
    )

    return {
        "adherence_score": score,
        "patterns_injected": total,
        "patterns_followed": followed,
        "patterns_violated": violated,
        "violations": violations,
        "detail": detail,
    }
