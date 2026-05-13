"""
Improvement validator — measures whether pattern learning is helping.

Compares two cohorts of drafts:
  BEFORE: drafts where applied_pattern_ids_json = "[]"  (no patterns injected)
  AFTER:  drafts where applied_pattern_ids_json has ≥1 pattern

For each cohort, computes:
  - avg grounding score
  - avg judge scores (groundedness, completeness, structure, overall)
  - draft count

Returns the delta so callers can report improvement (or regression).
"""

import json
import logging
from dataclasses import dataclass, field

from sqlmodel import Session, select

from python_service.db.models import Draft
from python_service.db.session import engine

logger = logging.getLogger(__name__)


@dataclass
class CohortStats:
    count: int = 0
    avg_grounding: float = 0.0
    avg_groundedness: float = 0.0
    avg_completeness: float = 0.0
    avg_structure: float = 0.0
    avg_overall: float = 0.0


@dataclass
class ImprovementReport:
    before: CohortStats = field(default_factory=CohortStats)
    after: CohortStats = field(default_factory=CohortStats)
    delta_grounding: float = 0.0
    delta_overall: float = 0.0
    has_data: bool = False
    message: str = ""


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _cohort_stats(drafts: list[Draft]) -> CohortStats:
    if not drafts:
        return CohortStats()

    groundings, gnd, comp, struct, overall = [], [], [], [], []

    for draft in drafts:
        groundings.append(draft.grounding_score)
        try:
            scores = json.loads(draft.judge_scores_json or "{}")
            if scores:
                gnd.append(float(scores.get("groundedness", 0)))
                comp.append(float(scores.get("completeness", 0)))
                struct.append(float(scores.get("structure", 0)))
                overall.append(float(scores.get("overall", 0)))
        except (json.JSONDecodeError, TypeError):
            pass

    return CohortStats(
        count=len(drafts),
        avg_grounding=_avg(groundings),
        avg_groundedness=_avg(gnd),
        avg_completeness=_avg(comp),
        avg_structure=_avg(struct),
        avg_overall=_avg(overall),
    )


def compute_improvement_report() -> ImprovementReport:
    """
    Load all drafts from SQLite, split into before/after cohorts,
    compute stats, and return the delta.
    """
    with Session(engine) as session:
        drafts = session.exec(select(Draft)).all()

    before_drafts, after_drafts = [], []
    for d in drafts:
        try:
            applied = json.loads(d.applied_pattern_ids_json or "[]")
        except json.JSONDecodeError:
            applied = []

        if applied:
            after_drafts.append(d)
        else:
            before_drafts.append(d)

    before = _cohort_stats(before_drafts)
    after = _cohort_stats(after_drafts)

    has_data = before.count > 0 and after.count > 0
    delta_grounding = round(after.avg_grounding - before.avg_grounding, 3)
    delta_overall = round(after.avg_overall - before.avg_overall, 3)

    if not has_data:
        message = (
            "Not enough data yet. Need drafts both before and after pattern learning. "
            "Run scripts/seed.py to pre-populate example patterns, then generate more drafts."
        )
    elif delta_overall >= 0:
        message = f"Pattern learning improved overall judge score by {delta_overall:+.2f} points."
    else:
        message = f"Overall score changed by {delta_overall:+.2f}. More edits needed to reinforce patterns."

    logger.info(
        "Improvement report: before=%d drafts (overall=%.2f) after=%d drafts (overall=%.2f) Δ=%+.2f",
        before.count, before.avg_overall,
        after.count, after.avg_overall,
        delta_overall,
    )

    return ImprovementReport(
        before=before,
        after=after,
        delta_grounding=delta_grounding,
        delta_overall=delta_overall,
        has_data=has_data,
        message=message,
    )
