"""
Adherence checker — verifies the draft followed the injected style patterns.

Phase C upgrade: uses NLI (DeBERTa) instead of substring search.

WHY upgrade from substring to NLI?
  The old checker asked: "does the wrong phrase appear in the draft?"
  That catches 'salary' but misses 'annual earnings' or 'total remuneration'.
  NLI asks: "does the draft TEXT ENTAIL this rule?"  The model understands
  that 'Base Compensation' and 'annual earnings' express the same concept
  and can flag violations regardless of phrasing.

  For NEUTRAL cases (model is unsure), we fall back to substring search so
  explicit few_shot_before phrases are still caught.

Adherence score = patterns_followed / patterns_injected
  1.0 = draft followed all injected rules
  0.0 = draft violated all injected rules
"""

import logging

logger = logging.getLogger(__name__)

# NLI context limit: DeBERTa handles ~512 tokens. We truncate the full draft
# to 4000 characters (roughly 600–800 tokens) to stay safely within the limit
# while capturing most of the draft.
_NLI_PREMISE_CHAR_LIMIT = 4000


def check_adherence(
    draft_sections: list[dict],
    injected_patterns: list[dict],
) -> dict:
    """
    Check whether the draft followed each injected pattern.

    Uses NLI (cross-encoder/nli-deberta-v3-small) to evaluate whether the
    draft text entails each pattern's rule. Falls back to substring search
    for NEUTRAL cases where the model is undecided.

    Args:
        draft_sections:    list of section dicts from Gemini output.
        injected_patterns: patterns that were injected into the prompt.

    Returns:
        Dict with adherence_score (0.0–1.0), followed/violated counts,
        and per-pattern detail including the NLI label.
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

    # Flatten all section content into one string for the NLI premise.
    # We truncate to stay within the model's token limit.
    full_text = " ".join(s.get("content", "") for s in draft_sections)
    premise = full_text[:_NLI_PREMISE_CHAR_LIMIT]

    # Lazy-load the NLI model (it's already cached as a singleton in verifier.py
    # from grounding verification, so this call is near-zero cost the 2nd time).
    try:
        from python_service.nli import verifier as nli
        nli_available = True
    except Exception as exc:
        logger.warning("NLI model unavailable — falling back to substring adherence: %s", exc)
        nli_available = False

    followed = 0
    violated = 0
    violations: list[str] = []
    detail: list[dict] = []

    for pattern in injected_patterns:
        description   = pattern.get("description", "")
        before_phrase = pattern.get("fewShotBefore", "").strip().lower()
        pattern_id    = pattern.get("pattern_id")

        if not description:
            followed += 1
            detail.append({
                "pattern_id":  pattern_id,
                "description": description,
                "result":      "UNCHECKED",
                "reason":      "no description to check",
            })
            continue

        # ── NLI check ────────────────────────────────────────────────────────
        # We ask: "does this draft (premise) ENTAIL this rule (hypothesis)?"
        # ENTAILMENT  → draft is consistent with the rule → FOLLOWED
        # CONTRADICTION → draft contradicts the rule → VIOLATED
        # NEUTRAL     → model is uncertain → fall back to substring
        nli_label = "NEUTRAL"
        if nli_available:
            try:
                nli_label = nli.predict(premise=premise, hypothesis=description)
            except Exception as exc:
                logger.warning(
                    "NLI predict failed for pattern %r: %s — using NEUTRAL fallback",
                    pattern_id, exc,
                )

        if nli_label == "ENTAILMENT":
            followed += 1
            detail.append({
                "pattern_id":  pattern_id,
                "description": description,
                "result":      "FOLLOWED",
                "nli_label":   nli_label,
            })

        elif nli_label == "CONTRADICTION":
            violated += 1
            violations.append(description)
            detail.append({
                "pattern_id":  pattern_id,
                "description": description,
                "result":      "VIOLATED",
                "nli_label":   nli_label,
            })
            logger.info("Pattern VIOLATED (NLI CONTRADICTION): %s", description[:60])

        else:
            # NEUTRAL: NLI is unsure. Fall back to substring search.
            # If the "wrong" before-phrase is present, the rule was likely violated.
            if before_phrase and before_phrase in full_text.lower():
                violated += 1
                violations.append(description)
                detail.append({
                    "pattern_id":   pattern_id,
                    "description":  description,
                    "result":       "VIOLATED",
                    "nli_label":    nli_label,
                    "found_phrase": pattern.get("fewShotBefore"),
                })
                logger.info(
                    "Pattern VIOLATED (NLI NEUTRAL + substring hit): %r in draft",
                    before_phrase[:50],
                )
            else:
                followed += 1
                detail.append({
                    "pattern_id":  pattern_id,
                    "description": description,
                    "result":      "FOLLOWED",
                    "nli_label":   nli_label,
                })

    total = len(injected_patterns)
    score = round(followed / total, 3) if total > 0 else 1.0

    logger.info(
        "Adherence (NLI): %d/%d patterns followed → score=%.2f",
        followed, total, score,
    )

    return {
        "adherence_score":    score,
        "patterns_injected":  total,
        "patterns_followed":  followed,
        "patterns_violated":  violated,
        "violations":         violations,
        "detail":             detail,
    }
