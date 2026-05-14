"""
Grounding verifier — checks every draft claim against the cited evidence.

For each sentence in the draft:
  - If it has a [En] citation → run NLI: does evidence[n] support this sentence?
  - If it has no citation but contains named entities → flag as UNGROUNDED_CLAIM

groundingScore = verified_sentences / total_checked_sentences

Three-tier fail-closed:
  >= 0.75 → HIGH   (deliver)
  >= 0.50 → MEDIUM (deliver with warnings)
  <  0.50 → LOW    (refuse — return diagnostic, don't deliver draft)
"""

import logging
import re
from dataclasses import dataclass, field

from python_service.nli import verifier as nli

logger = logging.getLogger(__name__)

# Regex to find [E1], [E2], ... citations in draft text
_CITATION_RE = re.compile(r"\[E(\d+)\]")

# Simple named-entity indicators — uncited sentences containing these are flagged
_ENTITY_PATTERNS = [
    re.compile(r"\$[\d,]+"),                    # dollar amounts
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),# dates MM/DD/YY
    re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+"),
    re.compile(r"\bSection\s+\d+", re.IGNORECASE),
    re.compile(r"\bArticle\s+[IVX\d]+", re.IGNORECASE),
]

GROUNDING_HIGH   = 0.75
GROUNDING_MEDIUM = 0.50


@dataclass
class GroundingWarning:
    warning_type: str    # HALLUCINATION_CANDIDATE | WEAK_GROUNDING | UNGROUNDED_CLAIM
    sentence: str
    evidence_id: str = ""
    nli_label: str = ""


@dataclass
class GroundingResult:
    grounding_score: float
    status: str                                  # HIGH | MEDIUM | LOW
    verified_count: int
    total_checked: int
    warnings: list[GroundingWarning] = field(default_factory=list)
    ungrounded_claims: list[str] = field(default_factory=list)
    diagnostic: str = ""


def verify_draft(
    draft_sections: list[dict],
    evidence_map: dict[str, str],   # {"E1": chunk_content, "E2": chunk_content, ...}
) -> GroundingResult:
    """
    Verify grounding of all sections in a draft.

    Args:
        draft_sections: list of section dicts from Gemini (each has "content").
        evidence_map:   dict mapping evidence IDs to their source text content.

    Returns:
        GroundingResult with score, status, and detailed warnings.
    """
    all_sentences: list[str] = []
    for section in draft_sections:
        content = section.get("content", "")
        sentences = _split_sentences(content)
        all_sentences.extend(sentences)

    if not all_sentences:
        return GroundingResult(
            grounding_score=0.0,
            status="LOW",
            verified_count=0,
            total_checked=0,
            diagnostic="Draft has no content to verify.",
        )

    # Build NLI batch: only sentences that have citations
    nli_pairs: list[tuple[str, str, str]] = []   # (evidence_id, premise, hypothesis)
    uncited_with_entities: list[str] = []

    for sentence in all_sentences:
        citations = _CITATION_RE.findall(sentence)
        if citations:
            for eid_num in citations:
                eid = f"E{eid_num}"
                premise = evidence_map.get(eid, "")
                if premise:
                    nli_pairs.append((eid, premise, sentence))
        else:
            if _has_entity(sentence):
                uncited_with_entities.append(sentence)

    # Run NLI in batch
    labels = nli.predict_batch([(p[1], p[2]) for p in nli_pairs])

    warnings: list[GroundingWarning] = []
    verified = 0
    total_checked = len(nli_pairs)

    contradictions = 0
    for (eid, premise, sentence), label in zip(nli_pairs, labels):
        if label == "ENTAILMENT":
            verified += 1
        elif label == "CONTRADICTION":
            contradictions += 1
            warnings.append(GroundingWarning(
                warning_type="HALLUCINATION_CANDIDATE",
                sentence=sentence,
                evidence_id=eid,
                nli_label=label,
            ))
        else:   # NEUTRAL — premise neither supports nor refutes; NOT verified.
            # Counting NEUTRAL as verified would inflate the grounding score on
            # vague paraphrases that don't actually entail the claim.
            warnings.append(GroundingWarning(
                warning_type="WEAK_GROUNDING",
                sentence=sentence,
                evidence_id=eid,
                nli_label=label,
            ))

    # Flag uncited sentences that contain named entities
    for sentence in uncited_with_entities:
        warnings.append(GroundingWarning(
            warning_type="UNGROUNDED_CLAIM",
            sentence=sentence,
        ))

    # Score = fraction not contradicted. NEUTRAL means "not refuted" — acceptable for
    # legal paraphrases. Only CONTRADICTION (direct conflict) counts against the score.
    score = (verified / total_checked) if total_checked > 0 else 1.0

    if score >= GROUNDING_HIGH:
        status = "HIGH"
    elif score >= GROUNDING_MEDIUM:
        status = "MEDIUM"
    else:
        status = "LOW"

    diagnostic = ""
    if status == "LOW":
        diagnostic = (
            f"Grounding score {score:.2f} is below the minimum threshold "
            f"({GROUNDING_MEDIUM}). Draft not delivered to prevent hallucination propagation."
        )

    logger.info(
        "Grounding: %d/%d verified → score=%.2f → %s",
        verified, total_checked, score, status,
    )

    return GroundingResult(
        grounding_score=round(score, 3),
        status=status,
        verified_count=verified,
        total_checked=total_checked,
        warnings=warnings,
        ungrounded_claims=uncited_with_entities,
        diagnostic=diagnostic,
    )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ? ! boundaries."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 10]


def _has_entity(sentence: str) -> bool:
    """Return True if the sentence contains a named entity that should be cited."""
    return any(pat.search(sentence) for pat in _ENTITY_PATTERNS)
