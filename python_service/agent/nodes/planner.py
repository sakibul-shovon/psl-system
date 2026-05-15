"""
python_service/agent/nodes/planner.py

The Planner node — the first node the LangGraph calls.

Its job: take the user's query and decompose it into 4–7 focused sections,
each with its own retrieval query.

WHY a separate planner step?
In the old system, one broad query ("summarize compensation and termination")
retrieved a single pool of top-5 chunks for the whole draft. That's like
sending one researcher to find everything. The planner splits this into focused
sub-tasks: one query per section. "Compensation" gets its own evidence hunt;
"Termination" gets its own. Each section's evidence is tighter and more relevant.

This node also:
  - Loads the document title + type from SQLite (needed by all downstream nodes)
  - Retrieves learned patterns ONCE here, so all parallel executors share them
    (we don't want 5 executor nodes each calling Qdrant independently)
"""

import json
import logging

from groq import Groq
from sqlmodel import Session

from python_service.agent.state import DraftingState, SectionPlan
from python_service.config import settings
from python_service.db.models import Document
from python_service.db.session import engine
from python_service.edit_loop.pattern_retriever import retrieve_patterns
from python_service.embedder import embed_one
from python_service.vector.qdrant_store import qdrant_store
from python_service.observability.langfuse_client import observe

logger = logging.getLogger(__name__)

# ── Prompt template ────────────────────────────────────────────────────────────
# Note: we use double {{ }} for literal braces inside an f-string.
# The model must return ONLY JSON — no markdown fences, no preamble.

_PLANNER_PROMPT = """\
You are a senior legal document analyst at Pearson Specter Litt.

Your task: decompose the requested draft into 4-7 sections. Each section must
have its own focused retrieval query so evidence can be gathered precisely.

QUERY: {query}
DOCUMENT TYPE: {document_type}
DRAFT TYPE: {draft_type}
DOCUMENT TITLE: {document_title}
{episodic_context}
Rules:
- 4 sections minimum, 7 maximum
- Each retrieval_query must be specific to THAT section only (not the whole query)
- target_length_words must be between 100 and 300
- section_id must be "sec_1", "sec_2", etc.

Return ONLY valid JSON, no markdown, no explanation:
{{
  "sections": [
    {{
      "section_id": "sec_1",
      "title": "string — section heading",
      "brief": "one sentence describing what this section covers",
      "retrieval_query": "focused query to retrieve evidence for THIS section only",
      "target_length_words": 150
    }}
  ]
}}
"""


def _retrieve_episodic_context(query: str, document_type: str) -> str:
    """
    Fetch the top-3 most similar past draft sessions from episodic memory
    and format them as a readable block for the planner prompt.

    WHY inject past sessions into the planner?
    The planner is choosing retrieval queries and section structure. Knowing
    that "last time you drafted a lease termination summary, the operator
    reworded the 'notice period' section heavily" tells the planner to focus
    its retrieval query on notice-period clauses — a signal no pattern captures.

    Returns an empty string if no memories exist yet (safe default for
    the first draft, before any feedback has been submitted).
    """
    try:
        embed_text = f"{query} | {document_type}"
        vector = embed_one(embed_text)
        memories = qdrant_store.search_episodic_memories(query_vector=vector, limit=3)
    except Exception as exc:
        logger.warning("Episodic memory retrieval failed (skipping): %s", exc)
        return ""

    if not memories:
        return ""

    lines = [
        "\nPRIOR SIMILAR SESSIONS (episodic memory — use for context, not as facts):"
    ]
    for i, m in enumerate(memories, start=1):
        p = m.get("payload", {})
        q = p.get("query", "")[:120]
        doc_type = p.get("document_type", "?")
        edit_count = p.get("edit_count", 0)
        dist = p.get("edit_distance_total", 0)
        grounding = p.get("grounding_score", 0.0)
        judge = p.get("judge_overall")
        judge_str = f"{judge:.1f}/5" if judge is not None else "unscored"
        lines.append(
            f"  [{i}] Query: \"{q}\" | DocType: {doc_type} | "
            f"Operator made {edit_count} edit(s), total distance {dist} | "
            f"Grounding: {grounding:.2f} | Judge: {judge_str}"
        )
    lines.append(
        "  → Sections that were heavily edited suggest the retrieval query "
        "needs to be more specific for those topics.\n"
    )
    return "\n".join(lines)


@observe(name="groq-planner")
def _call_gemini(prompt: str) -> dict:
    """Call Groq Llama 3.3 70B in JSON mode and return the parsed dict."""
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a legal document analyst. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=2048,
    )
    return json.loads(response.choices[0].message.content)


def planner_node(state: DraftingState) -> dict:
    """
    LangGraph node — takes the current state, returns a dict of fields to update.

    LangGraph merges the returned dict INTO the state automatically.
    We must NOT return the full state — only the fields we are adding/changing.

    What we add:
      - document_title  (looked up from SQLite)
      - document_type   (looked up from SQLite)
      - plan            (list of SectionPlan from Gemini)
      - patterns        (retrieved once, shared by all executors)
      - iteration       (initialised to 0 — the critic/refiner loop counter)
      - section_drafts  (initialised to empty list — executors will fill it)
    """
    document_id = state["document_id"]
    query       = state["query"]
    draft_type  = state.get("draft_type", "case_fact_summary")

    # ── Step 1: Load document metadata from SQLite ────────────────────────────
    with Session(engine) as session:
        doc = session.get(Document, document_id)
    if not doc:
        raise ValueError(f"Document {document_id!r} not found in database")

    document_title = doc.title
    document_type  = getattr(doc, "document_type", "unknown")
    logger.info("Planner: document=%r type=%r query=%r", document_title, document_type, query)

    # ── Step 2: Retrieve learned patterns once ────────────────────────────────
    # skip_patterns=True is set by scripts/ab_test.py for the control group.
    # In that case we return an empty list so the draft is generated without
    # any learned patterns — giving us the "no patterns" baseline for the A/B test.
    if state.get("skip_patterns", False):
        patterns = []
        logger.info("Planner: skip_patterns=True — returning empty pattern list (A/B control group)")
    else:
        patterns = retrieve_patterns(
            query=query,
            document_type=document_type,
            draft_type=draft_type,
        )
        logger.info("Planner: %d patterns retrieved", len(patterns))

    # ── Step 3: Retrieve episodic memories for context ───────────────────────
    # Embed "query | document_type", search for similar past sessions.
    # The formatted string is injected into the planner prompt so Gemini
    # can see how similar drafts were received by operators previously.
    episodic_context = _retrieve_episodic_context(query, document_type)
    logger.info(
        "Planner: episodic context %s",
        "found" if episodic_context else "empty (no prior sessions)",
    )

    # ── Step 4: Call Gemini to decompose the query into sections ──────────────
    prompt = _PLANNER_PROMPT.format(
        query=query,
        document_type=document_type,
        draft_type=draft_type,
        document_title=document_title,
        episodic_context=episodic_context,
    )

    try:
        result = _call_gemini(prompt)
        raw_sections = result.get("sections", [])
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Planner Gemini call failed: %s — using fallback plan", exc)
        # Fallback: one generic section so the pipeline can still continue
        raw_sections = [
            {
                "section_id":           "sec_1",
                "title":                "Document Summary",
                "brief":                "General summary of the document",
                "retrieval_query":      query,
                "target_length_words":  250,
            }
        ]

    # ── Step 5: Validate and coerce into SectionPlan TypedDicts ───────────────
    # Gemini might return slightly wrong field names or missing fields.
    # We apply safe defaults so downstream nodes never crash on a missing key.
    plan: list[SectionPlan] = []
    for i, s in enumerate(raw_sections[:7], start=1):   # cap at 7 sections
        plan.append(
            SectionPlan(
                section_id=s.get("section_id", f"sec_{i}"),
                title=s.get("title", f"Section {i}"),
                brief=s.get("brief", ""),
                retrieval_query=s.get("retrieval_query", query),
                target_length_words=max(100, min(300, int(s.get("target_length_words", 150)))),
            )
        )

    logger.info("Planner: produced %d-section plan: %s", len(plan), [p["title"] for p in plan])

    # Return only the fields we are setting — LangGraph merges these into state
    return {
        "document_title":  document_title,
        "document_type":   document_type,
        "plan":            plan,
        "patterns":        patterns,
        "iteration":       0,
        "section_drafts":  [],   # executors will append to this
    }
