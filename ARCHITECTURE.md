# PSL Document Intelligence — Architecture

This document is a technical deep-dive into how the PSL system works. It covers data flow, component design, latency profile, and the rationale behind key structural choices. For quick-start instructions see [README.md](README.md).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Pillar 1 — Ingestion Pipeline](#2-pillar-1--ingestion-pipeline)
3. [Pillar 2 — Hybrid Retrieval](#3-pillar-2--hybrid-retrieval)
4. [Pillar 3 — Agentic Draft (LangGraph)](#4-pillar-3--agentic-draft-langgraph)
5. [Pillar 4 — Learning Loop](#5-pillar-4--learning-loop)
6. [Episodic Memory](#6-episodic-memory)
7. [Observability Stack](#7-observability-stack)
8. [Data Model](#8-data-model)
9. [Latency Profile](#9-latency-profile)
10. [Failure Modes and Fallbacks](#10-failure-modes-and-fallbacks)

---

## 1. System Overview

The system is a Python monolith — one FastAPI process hosts all routes. Compute-heavy operations (OCR, embedding, reranking) run synchronously inside the request; ingestion runs as a FastAPI `BackgroundTask`. State is split between two stores:

- **SQLite** (via SQLModel/SQLAlchemy): relational metadata — documents, chunks, drafts, patterns, traces, episodic memories. Fast reads, single-writer.
- **Qdrant**: three vector collections for semantic search — `legal_chunks`, `learned_patterns`, `episodic_memory`. All 768-dimensional (BAAI/bge-base-en-v1.5).

There is no message queue. There is no microservice split. This is deliberate: for a system of this scale and a single-reviewer demo, the operational complexity of a distributed system would outweigh the benefit.

---

## 2. Pillar 1 — Ingestion Pipeline

### Entry point: `POST /upload`

A file arrives. The handler validates the extension, generates a UUID for the document, saves the file to `data/uploads/`, writes a minimal `Document` row to SQLite, and starts `run_ingestion_pipeline()` in the background.

### Route decision: `ingestion/file_router.py`

Every input file enters the same three-way router:

```
File
 ├── Contains text layer? (pypdf text extraction > 20 chars/page)
 │     └── TEXT_LAYER → skip OCR entirely, use extracted Unicode text
 ├── Tesseract confidence ≥ 0.35 on a sample page?
 │     └── TYPED_SCAN → Tesseract OCR
 └── Tesseract confidence < 0.35?
       └── HANDWRITING → TrOCR fallback
```

This routing is important because OCR on a text-layer PDF degrades the text (Tesseract can mis-read fonts at edges) and wastes 8–15 seconds per page. The `pypdf` text layer check runs in milliseconds.

### TrOCR handwriting fallback: `ocr/trocr_backend.py`

When Tesseract confidence is below 0.35 (the `TROCR_FALLBACK_THRESHOLD`), `microsoft/trocr-base-handwritten` (~400 MB, lazy-loaded on first use) takes over. The backend:

1. Loads the page as a grayscale image via pypdfium2.
2. Runs OpenCV morphological dilation to detect text-line bounding boxes.
3. Crops each bounding box and runs TrOCR inference per line.
4. Assigns confidence 0.60 (fixed) — below the `LOW_CONF_THRESHOLD` of 0.70, so every TrOCR line gets a `[LOW_CONF: 0.60]` annotation. This is honest: TrOCR is doing its best, but the system should not pretend handwritten text was cleanly read.

### Text normalisation: `ingestion/line_normalizer.py`

Tesseract output has characteristic noise patterns: hyphenation at line breaks (`compensa-\ntion`), inconsistent spacing around punctuation, ligature artefacts (`ﬁ` instead of `fi`). The normaliser runs a deterministic sequence of regex substitutions to clean these without risking information loss.

### Legal-structure chunking: `chunking/legal_chunker.py`

Legal documents have an internal hierarchy: Article → Section → Clause → Sub-clause. The chunker uses heading-pattern regex (e.g., `^(\d+\.)+\s`, `^ARTICLE\s+[IVXLCD]+`, `^Section\s+\d+`) to identify structural boundaries, then groups text under its nearest parent heading.

Each chunk gets a `breadcrumb` field: `"Article 4 → Section 4.2 → Clause 4.2(b)"`. This breadcrumb appears next to evidence in the generated draft and in the UI's expandable evidence viewer. It lets a lawyer navigate directly to the source.

Chunks are sized to fit within context: target 400–600 tokens, hard max 800 tokens. Chunks that hit the hard max are split at the nearest sentence boundary.

### Embedding and storage

Every chunk is embedded with `BAAI/bge-base-en-v1.5` (768-dim, 512 token context). Two embeddings are stored per chunk: content embedding (the full chunk text) and title embedding (the breadcrumb). The title embedding is used for section-header queries; both are stored as named vectors in Qdrant.

BM25 indexing uses `rank_bm25`. The index is serialised as a pickle file to `data/bm25/{document_id}.pkl`. BM25 requires the vocabulary at query time, which is why the chunk_ids must be reconstructed in the same order as the index was built (handled in `retrieval/hybrid.py:_get_ordered_chunk_ids()`).

---

## 3. Pillar 2 — Hybrid Retrieval

Every evidence retrieval passes through five sequential stages. The entry point is `retrieval/hybrid.py:retrieve()`.

### Stage 1: Dense search

`QdrantClient.query_points()` performs approximate nearest-neighbour search over the `legal_chunks` collection, filtered by `document_id`. Returns top-20 chunks by cosine similarity. Uses the content-vector named vector.

### Stage 2: BM25 keyword search

The pre-built BM25 index for the document is loaded from disk. A standard BM25 query (using Okapi BM25 with default k1=1.5, b=0.75) returns the top-20 chunks by keyword relevance.

BM25 is essential for legal documents. Terms like `"Section 4.2(b)"`, `"Base Compensation"`, or `"Date of Termination"` are defined terms that may not appear semantically similar to the query in embedding space, but are exact token matches. Dense search alone misses these.

### Stage 3: Reciprocal Rank Fusion (RRF)

`retrieval/rrf.py` merges the two ranked lists using the RRF formula: `score(d) = Σ 1 / (k + rank_i(d))` where k=60. This is parameter-free except for k, which controls how much early-rank advantage matters.

The merged list contains up to 20 unique chunks. Chunks that appeared in both lists get a significant RRF boost — that's the intent.

### Stage 4: Cross-encoder reranking

`retrieval/reranker.py` runs `cross-encoder/ms-marco-MiniLM-L-6-v2` over all (query, chunk_content) pairs from the fused list. The cross-encoder scores relevance jointly — it considers query and passage together, not as independent embeddings. This gives substantially more precise relevance scores than dense similarity.

The cross-encoder scores each pair in a single forward pass per pair. With 20 candidates, this is 20 inference calls in sequence — fast enough (~150 ms total on CPU) but not batched.

The re-ranked list is trimmed to top-5. These five become `[E1]` through `[E5]` in the generated draft.

### Stage 5: Sufficiency guard

```python
def is_sufficient(reranked: list[dict]) -> bool:
    if not reranked:
        return False
    return reranked[0]["rerank_score"] >= 0.35
```

If the best-scoring evidence chunk scores below 0.35, the pipeline declares insufficient evidence. This threshold was calibrated to reject genuinely off-topic queries while accepting semantically fuzzy but legitimate queries (e.g., "what happens if someone quits?" against a document that only uses "termination").

---

## 4. Pillar 3 — Agentic Draft (LangGraph)

The generation pipeline is a LangGraph `StateGraph` compiled to a singleton at module import time (`agent/graph.py`). The graph has six nodes and two routing functions. All sections of a document are drafted in parallel.

### State: `agent/state.py`

`DraftingState` is a TypedDict. The `section_drafts` field uses a custom reducer (`_merge_section_drafts`) that merges drafts by `section_id`, allowing parallel executor nodes to write to the same list without race conditions. LangGraph's fan-out/fan-in guarantees that the reducer is called atomically.

### Node: Planner

`agent/nodes/planner.py` calls Gemini 2.5 Flash once to decompose the user query into 4–7 `SectionPlan` objects. Each plan has a `title`, `brief`, `retrieval_query` (a focused sub-query specific to this section), and `target_length_words`.

The retrieval_query is crucial: instead of using the same query for all sections, each section gets a targeted sub-query. A query "summarize the employment agreement" decomposes into individual plans for compensation, termination, confidentiality, etc., each with a retrieval query designed to find evidence for that specific section.

The planner also receives **episodic context** — the three most similar past sessions from `episodic_memory` in Qdrant. This allows the planner to note, for example, "in previous sessions for employment agreements, sections on compensation required focused queries using the term 'Base Salary' rather than 'salary'".

### Node: Executor (runs N times in parallel)

`agent/nodes/executor.py` drafts exactly one section. It receives one `SectionPlan` from `current_section` in state and:

1. Runs focused hybrid retrieval using the section's own `retrieval_query`.
2. Formats evidence as `[E1]...[E5]` blocks for the Gemini prompt.
3. Injects up to 5 learned patterns (from the `patterns` list in state).
4. Calls Gemini 2.5 Flash (temperature=0.2, response_mime_type="application/json") for JSON output.
5. Runs NLI grounding on the generated section against its own evidence pool.
6. Returns a `SectionDraft` with `grounding_score`, `confidence`, `cited_evidence`, and `evidence_items`.

The `dispatch_to_executors` function in `dispatcher.py` uses LangGraph's `Send` primitive to fan all sections out simultaneously. All sections run in parallel — if the planner creates 5 sections, 5 executor calls run concurrently.

### Node: Critic

`agent/nodes/critic.py` inspects every completed `SectionDraft` and identifies "weak sections" using three criteria:

- `UNGROUNDED`: `grounding_score < 0.50` — less than half the claims are NLI-verified
- `INCOMPLETE`: content length < 40 words, or content contains `[INSUFFICIENT EVIDENCE`
- `STYLE_VIOLATION`: the NLI-based adherence checker finds a pattern explicitly VIOLATED (CONTRADICTION verdict)

The critic tracks `state["iteration"]` to enforce `MAX_ITERATIONS = 3`. If all sections pass (or we've hit the limit), routing goes to the Assembler. Otherwise it goes to the Refiner.

### Node: Refiner

`agent/nodes/refiner.py` asks Gemini for an improved `retrieval_query` for each weak section. The improved query is written back into the `SectionPlan` in state. The Dispatcher then re-runs executors for only the weak sections (using `dispatch_rewrites`, which calls `Send` only for the sections in `state["weak_sections"]`).

### Node: Assembler

`agent/nodes/assembler.py` runs after the critic approves all sections (or `MAX_ITERATIONS` is exhausted). It:

1. Sorts sections into plan order by `section_id`.
2. Averages per-section grounding scores into an overall score.
3. Runs `check_adherence()` (NLI-based pattern adherence check).
4. Calls `judge_draft()` (Groq Llama 3.3 70B, 4-dimension 1-5 scoring).
5. Saves a `Draft` row to SQLite.
6. Writes final fields to state for the API route to build its response.

### Trace capture

`main.py` wraps the entire `run_agent()` call in a `with trace.stage("agent_graph")` block. After the call, `_agent_node_detail(state)` extracts the node sequence, plan sections, iteration count, per-section grounding/confidence, and patterns injected. This detail is attached to the trace's `agent_graph` stage metadata. `GET /traces/{id}` surfaces it in the UI's Agent Trace page.

---

## 5. Pillar 4 — Learning Loop

### Edit capture: `edit_loop/capture.py`

`POST /feedback` accepts a `draft_id` and a list of `{section_id, original_text, edited_text}` objects. These are stored as `Edit` rows in SQLite with the current timestamp and `operator_id`. `process_edit()` is then called in a `BackgroundTask` for each edit.

### Edit classification: `edit_loop/classifier.py`

```python
@observe(name="groq-classify-edit")
def classify_edit(original: str, edited: str) -> dict:
    ...
```

Groq Llama 3.3 70B (temperature=0) classifies the edit into a structured schema:

```json
{
  "edit_type": "terminology | phrasing | citation | structure | omission",
  "scope": "word | phrase | sentence | paragraph",
  "confidence": 0.0–1.0,
  "rule": "generalised instruction (not specific to this document)"
}
```

Temperature=0 is critical here. The classifier is used to score *patterns* — patterns drive future drafts. Inconsistent classification would produce inconsistent patterns.

### Pattern extraction: `edit_loop/pattern_extractor.py`

A second Groq call takes the classifier output and generates a few-shot training example: a `before` snippet (original text) and `after` snippet (edited version), abstracted enough to apply to future documents.

### Deduplication: `edit_loop/processor.py`

Before inserting a new pattern, the system embeds the rule description and searches `learned_patterns` in Qdrant for cosine similarity ≥ 0.85. If found:
- `frequency += 1`
- `confidence = min(confidence + 0.05, 0.99)`
- `last_reinforced_at = now()`
- `operator_ids` list updated (multi-operator consensus)
- `operator_consensus = len(unique_operators) / max(frequency, 1)`

If not found, a new `Pattern` row is inserted into SQLite and upserted into Qdrant.

### DPO preference data: `data/preferences.jsonl`

Every processed edit emits one line to a JSONL file:

```json
{
  "edit_id": "...",
  "chosen": "edited text (what the operator preferred)",
  "rejected": "original text (what the model generated)",
  "context": "document type + query context",
  "timestamp": "..."
}
```

This file is the raw material for future DPO (Direct Preference Optimisation) fine-tuning. The system doesn't fine-tune — but it emits the data needed to do so.

### Pattern decay: `scripts/prune_patterns.py`

Patterns that are never reinforced go stale. The prune script (run manually or on a schedule):
- **Archives** patterns with `frequency=1` and `last_reinforced_at > 60 days` — nobody confirmed this pattern was useful.
- **Flags** patterns with `operator_consensus < 0.3` and `frequency ≥ 5` for human review — multiple operators disagree on this rule.

---

## 6. Episodic Memory

Episodic memory records the *context* of each drafting session so the Planner can learn from history. Each record in `episodic_memory` Qdrant collection stores:
- The embedding of `"query | document_type"` — what was asked, about what kind of document.
- The `grounding_score` and `judge_overall` from that session — how well it went.
- The `edit_distance_total` from subsequent feedback — how much the operator had to fix.

At planning time, the 3 most similar past sessions are retrieved and formatted as a block in the planner's prompt:

```
SIMILAR PAST SESSIONS (for guidance):
  1. [employment_contract] "summarize compensation" — grounding: 0.84, judge: 4.1, edits: 12
     → Note: queries mentioning "Base Salary" retrieved better evidence than "salary"
```

This is session-level few-shot learning: the system doesn't remember *content* from past documents (that would be a privacy problem) but it does remember *what retrieval strategies worked*.

---

## 7. Observability Stack

### Per-stage timing: `tracing.py`

`TraceBuilder` wraps each pipeline stage in a context manager that records wall-clock timing. Stages: `retrieval`, `pattern_retrieval`, `agent_graph`, `grounding`, `adherence_check`, `judge`. Results are stored in `Trace` / `TraceStage` rows in SQLite.

`GET /traces/{id}` returns the full trace, including the agent-node detail injected by `_agent_node_detail()` in main.py.

### Langfuse: `observability/langfuse_client.py`

The `@observe` decorator wraps every LLM call (Gemini, Groq) and local model call (classifier, extractor, judge). If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`, traces appear in the Langfuse dashboard. If not, `@observe` is a no-op — the app functions identically without keys.

---

## 8. Data Model

```
Document          — one per uploaded file
  ↓ 1:N
Chunk             — one per extracted text segment (200–800 tokens, legal-structure-aware)
  ↓ references
Draft             — one per /draft call; stores sections JSON + grounding + judge scores
  ↓ 1:N
Edit              — operator corrections to sections
  ↓ triggers
Pattern           — generalised rule extracted from one or more edits; has frequency/confidence
  ↓ 1:N
PatternVersion    — immutable snapshot on each update (for audit trail)

EpisodicMemory    — one per feedback session; records quality metrics for the planner
Trace             — one per /draft call; links to TraceStage rows for per-step timing
TraceStage        — one per pipeline stage; includes name, duration_ms, metadata JSON
```

SQLite table definitions live in `db/models.py` (SQLModel — typed Python classes that generate DDL).

---

## 9. Latency Profile

Measured on a modern laptop (Apple M2, 16 GB RAM), CPU-only inference.

| Stage | Typical latency | Notes |
|-------|----------------|-------|
| Ingestion (text-layer PDF, per page) | 1–3 s | Chunking + embedding |
| Ingestion (typed scan, per page) | 8–15 s | Adds Tesseract OCR |
| Ingestion (handwriting, per page) | 25–40 s | Adds TrOCR + OpenCV line detection |
| Dense retrieval (Qdrant, 1 document) | 20–50 ms | ANN search |
| BM25 retrieval | 10–30 ms | From pickled index |
| Cross-encoder reranking (20 candidates) | 100–200 ms | 20 inference calls |
| NLI grounding check (per sentence) | 40–80 ms | DeBERTa forward pass |
| Gemini generation (one section) | 2–5 s | Network-bound |
| Groq judge | 1–3 s | Network-bound |
| Full `/draft` (5 sections, no refinement) | 15–30 s | Executors run in parallel |
| Full `/draft` (with 1 refinement round) | 25–45 s | One extra parallel executor pass |

The dominant cost is LLM API calls. Local ML inference (embedder, NLI, reranker) adds ~500 ms total to a draft. Adding a GPU would reduce local inference to negligible, not improve total latency.

---

## 10. Failure Modes and Fallbacks

| Failure | Detection | Fallback |
|---------|-----------|----------|
| Qdrant unreachable | `qdrant_store.ensure_collections()` exception at startup | Warning logged; `/health` reflects status; pattern retrieval returns empty list; draft generates without patterns |
| Gemini API error (executor) | Try/except in `_call_gemini_for_section()` | Section gets content `[GENERATION ERROR: reason]`, grounding 0.0, confidence LOW |
| Groq API error (judge) | Try/except in `judge_draft()` | Returns default scores `{3, 3, 3, 3}` — draft saves but judge is flagged as unavailable |
| OCR confidence below threshold | Tesseract confidence < 0.35 | TrOCR fallback; if TrOCR also fails, Tesseract result kept with low-confidence annotations |
| Insufficient retrieval evidence | Reranker best score < 0.35 | `sufficient=False` → section content becomes `[INSUFFICIENT EVIDENCE: ...]` |
| Pattern extraction failure | Groq API error in processor | Edit is recorded in SQLite but no pattern is created; next feedback call will retry |
| Agent max iterations reached | `state["iteration"] >= MAX_ITERATIONS` | Assembler runs with whatever sections are passing; weak sections remain in output with LOW confidence |
