"""
Pattern quality gate — rejects low-quality patterns before they enter the store.

A pattern that passes gets saved to SQLite + embedded in Qdrant.
A pattern that fails gets the Edit marked as 'noise' and discarded.

Failure reasons are logged so we can tune the thresholds over time.

Thresholds:
  MIN_DESCRIPTION_WORDS = 6   — rules shorter than this are too vague
  MIN_CONFIDENCE        = 0.50 — extractor confidence below this is unreliable
  MIN_EXAMPLE_WORDS     = 3   — few_shot_before/after must be non-trivial
"""

import logging

logger = logging.getLogger(__name__)

MIN_DESCRIPTION_WORDS = 6
MIN_CONFIDENCE = 0.50
MIN_EXAMPLE_WORDS = 3

VALID_RULE_TYPES = {
    "terminology_change",
    "tone_shift",
    "citation_added",
    "fact_correction",
    "restructure",
    "omission_correction",
    "noise",
}


def passes_quality_gate(pattern: dict) -> tuple[bool, str]:
    """
    Validate a candidate pattern before persisting it.

    Args:
        pattern: dict from pattern_extractor.extract_pattern().

    Returns:
        (True, "") if the pattern is good.
        (False, reason) if it should be discarded.
    """
    if not pattern:
        return False, "pattern is None"

    description = pattern.get("description", "").strip()
    few_shot_before = pattern.get("few_shot_before", "").strip()
    few_shot_after = pattern.get("few_shot_after", "").strip()
    confidence = float(pattern.get("confidence", 0.0))
    rule_type = pattern.get("rule_type", "").lower()

    if not description:
        return False, "description is empty"

    desc_words = len(description.split())
    if desc_words < MIN_DESCRIPTION_WORDS:
        return False, f"description too short ({desc_words} words, min {MIN_DESCRIPTION_WORDS})"

    if confidence < MIN_CONFIDENCE:
        return False, f"confidence too low ({confidence:.2f}, min {MIN_CONFIDENCE})"

    if not few_shot_before or len(few_shot_before.split()) < MIN_EXAMPLE_WORDS:
        return False, "few_shot_before missing or too short"

    if not few_shot_after or len(few_shot_after.split()) < MIN_EXAMPLE_WORDS:
        return False, "few_shot_after missing or too short"

    if few_shot_before == few_shot_after:
        return False, "few_shot_before and few_shot_after are identical"

    if rule_type == "noise":
        return False, "rule_type is noise"

    logger.debug("Pattern passed quality gate: %s", description[:60])
    return True, ""
