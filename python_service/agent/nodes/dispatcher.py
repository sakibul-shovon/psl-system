"""
python_service/agent/nodes/dispatcher.py

The Dispatcher — a routing function that fans out to parallel Executor nodes.

This is NOT a node itself. It is a LangGraph "conditional edge" function
that runs between the Planner node and the Executor nodes. It reads the
section plan and fires off one Executor per section, all at the same time.

HOW LANGGRAPH PARALLELISM WORKS:
  A normal edge:  planner ──────────────► executor   (sequential, one call)

  A Send-based fan-out:
                            ┌──► executor (sec_1)
                            ├──► executor (sec_2)
    planner ──► dispatcher ─┼──► executor (sec_3)
                            ├──► executor (sec_4)
                            └──► executor (sec_5)

  LangGraph runs all five executor invocations in parallel (asyncio tasks).
  Each one gets its own copy of state with `current_section` set to its
  specific SectionPlan. Their results merge back via `operator.add`.

Send(node_name, state_patch):
  - `node_name` is the graph node to invoke ("executor")
  - `state_patch` is the dict merged into the current state for that invocation
  - We pass the full current state PLUS `current_section` set to one section

The dispatcher also handles the REFINER re-dispatch case (B.6):
  After the Critic flags weak sections, the Refiner needs to re-run the
  executor for just those weak sections. The same fan-out logic applies,
  just filtered to the weak subset.
"""

import logging

from langgraph.types import Send

from python_service.agent.state import DraftingState

logger = logging.getLogger(__name__)


def dispatch_to_executors(state: DraftingState) -> list[Send]:
    """
    Fan-out routing function: fires one Executor per section in the plan.

    Called as a conditional edge from the Planner node:
        graph.add_conditional_edges("planner", dispatch_to_executors)

    LangGraph calls this function, sees a list of Send objects, and
    schedules each one as an independent parallel invocation of "executor".

    Returns a list of Send — one per section plan item.
    """
    plan = state.get("plan", [])

    if not plan:
        # Safety net: planner produced nothing (shouldn't happen normally)
        logger.error("Dispatcher: empty plan — nothing to execute")
        return []

    logger.info(
        "Dispatcher: fanning out to %d executor(s): %s",
        len(plan),
        [s["title"] for s in plan],
    )

    # For each section, create a Send that:
    #  1. targets the "executor" node
    #  2. passes the current state PLUS current_section = this specific section
    #
    # {**state} copies all existing state fields (document_id, patterns, etc.)
    # so each executor has everything it needs without extra lookups.
    return [
        Send("executor", {**state, "current_section": section})
        for section in plan
    ]


def dispatch_rewrites(state: DraftingState) -> list[Send]:
    """
    Fan-out routing function for the REFINER path.

    Called after the Critic identifies weak sections. Only re-dispatches
    executors for sections flagged in the critique — not the whole plan.

    A weak section is one whose section_id appears in the critique list.
    The Refiner will have already updated `current_section` in the plan
    with an improved retrieval query; here we just fire those sections again.

    Called as a conditional edge from the Refiner node (B.6):
        graph.add_conditional_edges("refiner", dispatch_rewrites)
    """
    critique   = state.get("critique", [])
    plan       = state.get("plan", [])

    # Build a set of section_ids that need rewriting
    weak_ids = {c["section_id"] for c in critique}

    # Find the SectionPlan entries for the weak sections
    # (the plan may have been updated by the Refiner with better queries)
    sections_to_redo = [s for s in plan if s["section_id"] in weak_ids]

    if not sections_to_redo:
        logger.info("dispatch_rewrites: no weak sections — skipping re-execution")
        return []

    logger.info(
        "dispatch_rewrites: re-running %d section(s): %s",
        len(sections_to_redo),
        [s["title"] for s in sections_to_redo],
    )

    return [
        Send("executor", {**state, "current_section": section})
        for section in sections_to_redo
    ]
