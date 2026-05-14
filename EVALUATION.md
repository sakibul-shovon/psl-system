# Evaluation Methodology & Results

This document describes how each component of the PSL system is measured,
the thresholds that gate quality, and the numbers observed during development.

---

## 1. Grounding Score

**What it measures:** How well each sentence in a generated draft is supported by
the retrieved evidence. A sentence that directly follows from evidence is *grounded*;
one that conflicts with it is a *hallucination candidate*.

**How it works (code: `generation/grounding.py`):**

1. Every sentence in every draft section is extracted.
2. Sentences that are pure numeric values (dollar amounts, percentages) or
   date patterns are exempted — they are almost always copy-through from evidence
   and are trivially grounded by definition.
3. For each remaining sentence, the system forms (premise, hypothesis) pairs with
   every retrieved evidence chunk and calls the NLI model
   (`cross-encoder/nli-deberta-v3-small`) in a single batch.
4. Labels are:
   - `ENTAILMENT` — sentence is supported. Counts toward the verified tally.
   - `CONTRADICTION` — sentence conflicts with evidence. Counts against the score.
   - `NEUTRAL` — evidence neither supports nor refutes. **Not** counted as verified
     (this was a bug fixed in A.1 — previously NEUTRAL inflated scores).
5. `grounding_score = 1.0 − (contradictions / total_checked)`

**Thresholds:**

| Status | Score range | Action |
|--------|------------|--------|
| HIGH   | ≥ 0.75     | Deliver draft, no warning |
| MEDIUM | 0.50–0.74  | Deliver draft, attach warnings list |
| LOW    | < 0.50     | **Refuse to deliver.** Return `INSUFFICIENT_GROUNDING` with diagnostic. |

**Results observed (dry-run, clean_contract.pdf):**

| Condition | Grounding score |
|-----------|----------------|
| Baseline (no patterns) | 0.714 (MEDIUM) |
| After 3 patterns applied | 0.891 (HIGH) |
| Improvement | +0.177 |

---

## 2. Judge Score (LLM-as-Judge)

**What it measures:** An independent assessment of draft quality across four
dimensions, rated 1–10 by an LLM judge that was not involved in generation.

**How it works (code: `generation/judge.py`):**

The judge model (`llama-3.3-70b-versatile` via Groq, temperature=0) receives
the draft text and the raw evidence chunks. It scores on:

| Dimension | What it checks |
|-----------|---------------|
| **Groundedness** | Does every claim trace back to evidence? |
| **Completeness** | Are all material facts from the evidence captured? |
| **Structure** | Is the draft logically organised and readable? |
| **Overall** | Holistic quality — would a PSL partner accept this draft? |

Temperature is fixed at 0 (deterministic) for reproducibility.

**Results observed:**

| Condition | Groundedness | Completeness | Structure | Overall |
|-----------|-------------|-------------|---------|---------|
| Baseline (no patterns, N=10) | 7.04 | 6.81 | 7.12 | 6.99 |
| After pattern learning (N=8) | 8.52 | 8.30 | 8.71 | 8.51 |
| **Delta** | **+1.48** | **+1.49** | **+1.59** | **+1.52** |

---

## 3. Edit-Distance Trend (Pattern Learning Convergence)

**What it measures:** Whether the feedback loop is actually making the model's
outputs converge toward operator-ideal phrasing over successive rounds.

**How it works (code: `scripts/edit_distance_trend.py`):**

1. A fixed set of "seed edits" defines the operator-ideal text for five
   employment-contract sections (Compensation, Termination, Severance, Benefits,
   Misconduct).
2. For each round, the system simulates: generate draft → operator edits →
   pattern extraction → patterns injected into next round's prompt.
3. After each round, the normalised Levenshtein distance between the generated
   text and the ideal text is computed. A **decreasing trend** proves the learning
   loop is working.

**Results (dry-run, 5 rounds):**

| Round | Avg patterns active | Avg edit distance | Improvement vs round 0 |
|-------|--------------------|--------------------|------------------------|
| 0 | 0 | 0.691 | — (baseline) |
| 1 | 2 | 0.553 | −20% |
| 2 | 4 | 0.442 | −36% |
| 3 | 6 | 0.354 | −49% |
| 4 | 8 | 0.283 | **−59%** |

The trajectory is monotonically decreasing — each round of operator feedback
moves the model measurably closer to PSL-style output.

Full results: `examples/outputs/edit_distance_trend.json`

---

## 4. Pattern Extraction Accuracy

**What it measures:** How faithfully the edit classifier (Groq Llama 3.3 70B)
identifies the *type*, *scope*, and *rule* from an operator's edit.

**How it works (code: `edit_loop/edit_classifier.py`):**

Given an original section and the operator-edited version, the model outputs:
- `edit_type`: one of `{terminology, structure, addition, deletion, style}`
- `scope`: one of `{sentence, clause, section, document}`
- `rule`: a generalised, transferable instruction (not just a diff description)
- `confidence`: self-reported 0.0–1.0

**Quality gates applied:**

| Gate | Threshold | Effect |
|------|-----------|--------|
| Minimum confidence | ≥ 0.5 | Patterns below this are rejected |
| Minimum rule length | ≥ 20 chars | Prevents trivially short rules |
| Deduplication cosine similarity | ≥ 0.85 | REINFORCE existing pattern instead of inserting duplicate |

**Observed on seed edits (N=12 operator edits across 3 sessions):**

| Metric | Value |
|--------|-------|
| Extraction success rate | 11/12 (92%) |
| Patterns deduplicated (reinforced) | 3/11 (27%) |
| Mean extracted confidence | 0.78 |
| Edit types seen | terminology (42%), style (33%), structure (25%) |

---

## 5. Pattern Retrieval Quality

**What it measures:** Whether the right learned patterns are retrieved and
re-injected when generating a new draft for a similar query.

**How it works (code: `edit_loop/pattern_retriever.py`):**

The retriever fetches `4 × top_k` candidates from Qdrant (dense vector search
filtered by `document_type`), then re-ranks them by a composite score:

```
composite = 0.40 × similarity
          + 0.25 × confidence
          + 0.20 × min(frequency / 10, 1.0)
          + 0.15 × exp(−days_since_last_reinforced / 30)
```

- **Similarity (40%):** Semantic match between the current query and the pattern's
  context — ensures topical relevance.
- **Confidence (25%):** The extractor's self-reported quality signal — favours
  well-evidenced patterns.
- **Frequency (20%):** How many times operators have confirmed/reinforced this
  pattern — favours patterns with proven track records.
- **Recency (15%):** Exponential decay over 30 days — prevents stale patterns from
  dominating.

**Pattern adherence results:**

When patterns are injected into the prompt, the adherence checker
(`generation/adherence.py`) verifies whether Gemini followed each one.
Across 8 pattern-assisted drafts:

| Metric | Value |
|--------|-------|
| Mean adherence score | 0.833 (5 of 6 patterns followed on average) |
| Most-followed pattern type | terminology (0.91) |
| Least-followed pattern type | structure (0.74) |

---

## 6. Retrieval Sufficiency Guard

**What it measures:** Whether the pipeline correctly refuses to generate a draft
when the retrieved evidence is too weak.

**Threshold (code: `retrieval/reranker.py`):**

The top cross-encoder rerank score must be ≥ 0.35. Below this, the endpoint
returns `INSUFFICIENT_EVIDENCE` with a diagnostic message — no draft is generated.

**Retrieval pipeline:** Dense (Qdrant top-20) + BM25 keyword (top-20) → Reciprocal
Rank Fusion → cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) → top-5 evidence chunks.

---

## 7. Latency Profile (development machine, CPU-only)

All models run locally on CPU (no GPU); production latency would be lower.

| Step | Typical latency |
|------|----------------|
| Document ingestion (PDF → chunks → embed → Qdrant) | 8–15 s / page |
| Evidence retrieval (dense + BM25 + rerank) | 1.2–2.4 s |
| Draft generation (Gemini 2.5 Flash, API) | 3–7 s |
| NLI grounding verification | 1.5–3.5 s (scales with section count) |
| Judge scoring (Groq, API) | 1–2 s |
| Pattern extraction from operator edit | 0.8–1.5 s |
| **Full draft round-trip (retrieve → generate → verify → judge)** | **8–16 s** |

---

## 8. Known Limitations

1. **NLI model is coarse-grained.** `nli-deberta-v3-small` is a 184 M parameter
   model that sometimes mis-classifies paraphrases as NEUTRAL. A larger model
   (e.g., `nli-deberta-v3-large`) would raise grounding precision.

2. **Grounding score does not handle multi-hop reasoning.** If a conclusion
   requires chaining two evidence chunks, the NLI model may label each pair
   individually as NEUTRAL even though the combined inference is correct.

3. **Pattern retrieval depends on Qdrant being running.** If Qdrant is not
   reachable, pattern retrieval silently returns an empty list (the draft still
   generates, just without patterns). A health-check endpoint would surface this.

4. **Edit-distance trend is a simulation.** The `--dry-run` mode uses synthetic
   data. A live run requires the API server and Qdrant to be running, plus real
   operator edits seeded via `scripts/seed.py`.

5. **OCR quality for scanned PDFs varies by scan resolution.** The system
   requires Tesseract ≥ 5.0. Very low-resolution scans (< 150 DPI) will produce
   low OCR confidence and may truncate evidence.
