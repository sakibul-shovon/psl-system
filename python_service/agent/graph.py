"""
python_service/agent/graph.py

Wires all agent nodes into a LangGraph StateGraph and compiles it.

The graph replaces the old linear POST /draft pipeline with a
planner → parallel executors → critic → refiner loop → assembler structure.

Full data flow:

  START
    │
    ▼
  planner          (one Gemini call: query → section plan + load patterns)
    │
    │ conditional edge: dispatch_to_executors()
    │ returns [Send("executor", state+sec_1), Send("executor", state+sec_2), ...]
    │
    ├──► executor(sec_1) ─┐
    ├──► executor(sec_2) ─┤  (all run in parallel; results merge via reducer)
    ├──► executor(sec_3) ─┤
    ├──► executor(sec_4) ─┤
    └──► executor(sec_5) ─┘
                          │
                          ▼
                        critic     (reviews every SectionDraft)
                          │
                          │ conditional edge: should_refine()
                          │
              ┌───"assembler"──────────────────────────────────────┐
              │                                                     ▼
         "refiner"                                              assembler
              │                                                     │
              ▼                                                   END
           refiner   (improves retrieval queries for weak sections)
              │
              │ conditional edge: dispatch_rewrites()
              │ returns [Send("executor", ...)] for weak sections only
              │
              ├──► executor(sec_2) ─┐  (only the weak ones, in parallel)
              └──► executor(sec_4) ─┘
                                    │
                                    ▼
                                  critic   (loop back, iteration + 1)
                                    ...    (max 3 iterations then → assembler)

The `compile()` call validates the graph structure and returns an object
with `.invoke()` and `.ainvoke()` methods.
"""

import logging

from langgraph.graph import END, START, StateGraph

from python_service.agent.nodes.assembler  import assembler_node
from python_service.agent.nodes.critic     import critic_node, should_refine
from python_service.agent.nodes.dispatcher import dispatch_rewrites, dispatch_to_executors
from python_service.agent.nodes.executor   import executor_node
from python_service.agent.nodes.planner    import planner_node
from python_service.agent.nodes.refiner    import refiner_node
from python_service.agent.state            import DraftingState

logger = logging.getLogger(__name__)


def build_graph():
    """
    Construct and compile the drafting agent StateGraph.

    Returns a compiled LangGraph object ready for .invoke() or .ainvoke().
    Called once at startup; the result is module-level cached in `drafting_agent`.
    """
    graph = StateGraph(DraftingState)

    # ── Register nodes ────────────────────────────────────────────────────────
    # Each string name here is what Send() and add_edge() reference.
    graph.add_node("planner",   planner_node)
    graph.add_node("executor",  executor_node)
    graph.add_node("critic",    critic_node)
    graph.add_node("refiner",   refiner_node)
    graph.add_node("assembler", assembler_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "planner")

    # ── Planner → Executors (fan-out) ─────────────────────────────────────────
    # dispatch_to_executors() returns a list of Send objects — one per section.
    # LangGraph fires them in parallel and waits for ALL to finish before
    # moving any one of them to the next edge.
    graph.add_conditional_edges("planner", dispatch_to_executors)

    # ── Executors → Critic (fan-in) ───────────────────────────────────────────
    # After every executor (whether from the initial dispatch or a re-run)
    # finishes, the result flows into the critic.
    # LangGraph's fan-in: waits for ALL pending Send completions before
    # invoking critic with the merged state.
    graph.add_edge("executor", "critic")

    # ── Critic → Refiner or Assembler (conditional) ───────────────────────────
    # should_refine() returns "refiner" or "assembler" based on:
    #   - are there any weak sections in the critique?
    #   - have we hit MAX_ITERATIONS?
    graph.add_conditional_edges(
        "critic",
        should_refine,
        {"refiner": "refiner", "assembler": "assembler"},
    )

    # ── Refiner → Executors (fan-out, weak sections only) ────────────────────
    # dispatch_rewrites() returns Send objects only for weak sections.
    # Their results overwrite the old bad drafts via _merge_section_drafts.
    graph.add_conditional_edges("refiner", dispatch_rewrites)

    # ── Assembler → END ───────────────────────────────────────────────────────
    graph.add_edge("assembler", END)

    compiled = graph.compile()
    logger.info("Drafting agent graph compiled successfully")
    return compiled


# ── Module-level singleton ────────────────────────────────────────────────────
# Built once when this module is first imported.
# The API route imports `drafting_agent` and calls drafting_agent.invoke().
drafting_agent = build_graph()


def run_agent(
    document_id: str,
    query: str,
    draft_type: str = "case_fact_summary",
    skip_patterns: bool = False,
) -> DraftingState:
    """
    Entry point for the API route.

    Runs the full agent synchronously and returns the final DraftingState.
    The caller reads state["final_*"] fields to build the HTTP response.

    Args:
        document_id:   UUID of the ingested document in SQLite.
        query:         The user's natural-language request.
        draft_type:    One of "case_fact_summary", "demand_letter", etc.
        skip_patterns: If True, the planner skips pattern retrieval. Used by
                       scripts/ab_test.py to create a no-pattern control group.

    Returns:
        The completed DraftingState after the graph has run to END.
        Key fields: final_draft_id, final_sections, final_grounding_score,
                    final_judge_scores, final_adherence.
    """
    initial_state: DraftingState = {
        "document_id":   document_id,
        "query":         query,
        "draft_type":    draft_type,
        "skip_patterns": skip_patterns,
    }

    logger.info(
        "Agent run: document=%r query=%r draft_type=%r",
        document_id, query, draft_type,
    )

    final_state = drafting_agent.invoke(initial_state)

    logger.info(
        "Agent run complete: draft=%r grounding=%.3f sections=%d",
        final_state.get("final_draft_id", "?"),
        final_state.get("final_grounding_score", 0.0),
        len(final_state.get("final_sections", [])),
    )

    return final_state
