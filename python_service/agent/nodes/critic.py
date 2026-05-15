"""
python_service/agent/nodes/critic.py

The Critic node — reviews all section drafts and flags weak ones.

Runs AFTER all Executor nodes finish. By that point, state["section_drafts"]
contains one SectionDraft per section (all merged via operator.add).

The Critic is intentionally fast — it uses the grounding scores already
computed by each Executor (no extra LLM call for the basic checks) and
adds a lightweight LLM pass only for completeness + style issues that
grounding scores cannot detect.

Three weakness types it can flag:
  UNGROUNDED      — grounding_score < 0.50  (NLI says content conflicts with evidence)
  INCOMPLETE      — section is suspiciously short OR contains [INSUFFICIENT EVIDENCE]
  STYLE_VIOLATION — a learned pattern was injected but the section ignored it

After flagging, the graph checks whether to loop to the Refiner or move on.
That routing decision lives here too: `should_refine()` is the conditional
edge function that reads the critique and decides the next node.

MAX_ITERATIONS = 3 prevents infinite loops. After 3 refinement rounds,
the graph moves on with whatever quality it has achieved.
"""

import logging

from python_service.agent.state import CritiqueItem, DraftingState, SectionDraft
from python_service.evaluation.adherence_checker import check_adherence

logger = logging.getLogger(__name__)

# A section must score at least this to pass the grounding gate.
# Matches the MEDIUM threshold in generation/grounding.py.
GROUNDING_PASS_THRESHOLD = 0.50

# A section shorter than this (in words) is flagged as likely incomplete.
MIN_WORDS = 40

# How many refinement iterations to allow before giving up and moving on.
MAX_ITERATIONS = 3


def _count_words(text: str) -> int:
    return len(text.split())


def _check_grounding(draft: SectionDraft) -> CritiqueItem | None:
    """Flag a section whose NLI grounding score is below the pass threshold."""
    if draft["grounding_score"] < GROUNDING_PASS_THRESHOLD:
        return CritiqueItem(
            section_id=draft["section_id"],
            weakness_type="UNGROUNDED",
            description=(
                f"Grounding score {draft['grounding_score']:.2f} is below "
                f"the minimum {GROUNDING_PASS_THRESHOLD}. "
                f"The NLI model found the content conflicts with or is not "
                f"supported by the retrieved evidence."
            ),
            suggested_fix=(
                "Re-retrieve evidence with a broader query. "
                "Ensure every claim cites a specific evidence item. "
                "Remove any sentence that cannot be directly traced to evidence."
            ),
        )
    return None


def _check_completeness(draft: SectionDraft) -> CritiqueItem | None:
    """Flag a section that is too short or explicitly marked insufficient."""
    content = draft["content"]

    if "[INSUFFICIENT EVIDENCE" in content:
        return CritiqueItem(
            section_id=draft["section_id"],
            weakness_type="INCOMPLETE",
            description=(
                f"Section '{draft['title']}' contains an [INSUFFICIENT EVIDENCE] "
                f"marker — the executor could not find relevant evidence."
            ),
            suggested_fix=(
                "Re-retrieve with a simpler, broader query. "
                "Try synonyms or parent-level section headings as the search query."
            ),
        )

    word_count = _count_words(content)
    if word_count < MIN_WORDS:
        return CritiqueItem(
            section_id=draft["section_id"],
            weakness_type="INCOMPLETE",
            description=(
                f"Section '{draft['title']}' is only {word_count} words — "
                f"likely missing key information."
            ),
            suggested_fix=(
                "Retrieve more evidence chunks. "
                "Expand the retrieval query to include related terms."
            ),
        )

    return None


def _check_style(
    draft: SectionDraft,
    patterns: list[dict],
) -> CritiqueItem | None:
    """
    Flag a section that ignored injected style patterns.

    Reuses the existing adherence_checker — it does fast substring search
    for the few_shot_before phrases that operators said to avoid.
    """
    if not patterns:
        return None

    # adherence_checker expects section dicts with a "content" key
    section_as_dict = {
        "section_id": draft["section_id"],
        "title":      draft["title"],
        "content":    draft["content"],
    }
    result = check_adherence([section_as_dict], patterns)

    violations = result.get("violations", [])
    if violations:
        # adherence_checker returns violations as a list[str] (description strings)
        descriptions = "; ".join(str(v) for v in violations[:3])
        return CritiqueItem(
            section_id=draft["section_id"],
            weakness_type="STYLE_VIOLATION",
            description=(
                f"Section '{draft['title']}' violated {len(violations)} learned "
                f"style pattern(s): {descriptions}"
            ),
            suggested_fix=(
                "Rewrite this section following the style patterns exactly. "
                "Pay attention to preferred terminology and sentence structure."
            ),
        )

    return None


# ── Main node ─────────────────────────────────────────────────────────────────

def critic_node(state: DraftingState) -> dict:
    """
    LangGraph node — reviews all section drafts and returns a critique list.

    Returns:
      {"critique": [list of CritiqueItem], "iteration": current_count}

    An empty critique list means all sections passed — the graph will route
    to the assembler. A non-empty list triggers the refiner.
    """
    drafts   = state.get("section_drafts", [])
    patterns = state.get("patterns", [])
    iteration = state.get("iteration", 0)

    logger.info(
        "Critic (iteration %d): reviewing %d section(s)", iteration, len(drafts)
    )

    critique: list[CritiqueItem] = []

    for draft in drafts:
        # Run all three checks; take the FIRST (most severe) issue per section.
        # We only flag one weakness per section to keep the suggested_fix actionable.
        issue = (
            _check_grounding(draft)
            or _check_completeness(draft)
            or _check_style(draft, patterns)
        )
        if issue:
            critique.append(issue)
            logger.info(
                "Critic flagged [%s] %r: %s",
                draft["section_id"], draft["title"], issue["weakness_type"],
            )
        else:
            logger.info(
                "Critic approved [%s] %r (grounding=%.2f, words=%d)",
                draft["section_id"], draft["title"],
                draft["grounding_score"], _count_words(draft["content"]),
            )

    logger.info(
        "Critic summary: %d/%d sections need refinement",
        len(critique), len(drafts),
    )

    return {
        "critique":  critique,
        "iteration": iteration + 1,
    }


# ── Routing function (conditional edge) ───────────────────────────────────────

def should_refine(state: DraftingState) -> str:
    """
    Conditional edge function — called by LangGraph after the Critic node.

    Returns the name of the NEXT node to visit:
      "refiner"   — if there are weak sections AND we haven't hit the limit
      "assembler" — if everything passed OR we've exhausted our retry budget

    This function is registered as a conditional edge in the StateGraph:
        graph.add_conditional_edges("critic", should_refine)
    """
    critique  = state.get("critique", [])
    iteration = state.get("iteration", 0)

    if not critique:
        logger.info("Critic: all sections passed — routing to assembler")
        return "assembler"

    if iteration >= MAX_ITERATIONS:
        logger.info(
            "Critic: %d weak section(s) remain but max iterations (%d) reached "
            "— routing to assembler with best-effort output",
            len(critique), MAX_ITERATIONS,
        )
        return "assembler"

    logger.info(
        "Critic: %d weak section(s) — routing to refiner (iteration %d/%d)",
        len(critique), iteration, MAX_ITERATIONS,
    )
    return "refiner"
