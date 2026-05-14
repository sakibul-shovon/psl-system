"""
python_service/agent/nodes/refiner.py

The Refiner node — improves retrieval queries for weak sections so the
Executor can try again with better-targeted evidence.

This node does NOT rewrite the draft text itself. Instead it updates the
SectionPlan entries for weak sections with smarter retrieval_query values,
then hands off to dispatch_rewrites (B.4) which re-fires Executor for
each of those sections.

WHY update the query rather than the content?
  The root cause of a weak section is almost always bad retrieval — wrong
  chunks, too-narrow query, missing synonyms. Rewriting the text without
  better evidence just polishes a bad foundation. Better to go back and
  get the right evidence first.

The Refiner uses a small Gemini call to generate an improved query for
each weak section based on the CritiqueItem's weakness_type and description.
For STYLE_VIOLATION issues it also injects the specific pattern rule into
the executor's next prompt via a hint stored in the plan's brief field.
"""

import json
import logging

import google.generativeai as genai

from python_service.agent.state import CritiqueItem, DraftingState, SectionPlan
from python_service.config import settings

logger = logging.getLogger(__name__)

_REFINE_PROMPT = """\
A legal draft section was flagged as weak. Produce an improved retrieval query
so a vector search can find better evidence to rewrite this section.

SECTION TITLE: {title}
ORIGINAL RETRIEVAL QUERY: {original_query}
WEAKNESS TYPE: {weakness_type}
WEAKNESS DESCRIPTION: {description}
SUGGESTED FIX: {suggested_fix}

Rules:
- Return ONLY valid JSON, no markdown
- The new query must be different from the original — use synonyms or parent terms
- Keep it under 20 words

{{"improved_query": "your improved retrieval query here"}}
"""


def _improve_query(
    plan_section: SectionPlan,
    critique: CritiqueItem,
) -> str:
    """
    Ask Gemini for a better retrieval query for this weak section.

    Falls back to a simple broadening heuristic if the API call fails,
    so the pipeline never stalls on a refiner error.
    """
    if not settings.gemini_api_key:
        # Heuristic fallback: just append "clause terms provisions"
        return plan_section["retrieval_query"] + " clause terms provisions details"

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.3,
            max_output_tokens=256,
        ),
    )

    prompt = _REFINE_PROMPT.format(
        title=plan_section["title"],
        original_query=plan_section["retrieval_query"],
        weakness_type=critique["weakness_type"],
        description=critique["description"],
        suggested_fix=critique["suggested_fix"],
    )

    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        improved = result.get("improved_query", "").strip()
        if improved and improved != plan_section["retrieval_query"]:
            return improved
    except Exception as exc:
        logger.warning("Refiner query improvement failed for [%s]: %s", plan_section["section_id"], exc)

    # Fallback: append broader terms
    return plan_section["retrieval_query"] + " clause terms provisions details"


def refiner_node(state: DraftingState) -> dict:
    """
    LangGraph node — updates retrieval queries for weak sections.

    Reads:
      state["critique"]  — list of CritiqueItems from the Critic
      state["plan"]      — current list of SectionPlans

    Writes back:
      {"plan": updated_plan_with_better_queries}

    After this node, the graph calls dispatch_rewrites (B.4) which re-fires
    Executor nodes for only the weak sections. Those Executors use the new
    queries and their results overwrite the old bad drafts via the custom
    _merge_section_drafts reducer.
    """
    critique = state.get("critique", [])
    plan     = state.get("plan", [])
    iteration = state.get("iteration", 0)

    if not critique:
        logger.info("Refiner: no critique items — nothing to refine")
        return {}

    # Build a lookup: section_id -> CritiqueItem for fast access
    critique_by_id: dict[str, CritiqueItem] = {
        c["section_id"]: c for c in critique
    }

    logger.info(
        "Refiner (iteration %d): improving queries for %d weak section(s): %s",
        iteration,
        len(critique),
        [c["section_id"] for c in critique],
    )

    updated_plan: list[SectionPlan] = []
    for section in plan:
        sid = section["section_id"]
        if sid not in critique_by_id:
            # This section was fine — keep its plan unchanged
            updated_plan.append(section)
            continue

        crit = critique_by_id[sid]
        old_query = section["retrieval_query"]
        new_query = _improve_query(section, crit)

        logger.info(
            "Refiner [%s]: query updated\n  old: %r\n  new: %r",
            sid, old_query, new_query,
        )

        # For STYLE_VIOLATION, also update the brief to remind the executor
        # about the specific pattern it must follow this time
        new_brief = section["brief"]
        if crit["weakness_type"] == "STYLE_VIOLATION":
            new_brief = (
                section["brief"]
                + f" IMPORTANT: {crit['suggested_fix']}"
            )

        updated_plan.append(
            SectionPlan(
                section_id=sid,
                title=section["title"],
                brief=new_brief,
                retrieval_query=new_query,
                target_length_words=section["target_length_words"],
            )
        )

    return {"plan": updated_plan}
