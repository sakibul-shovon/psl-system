"""
scripts/judge_tournament.py

LLM judge tournament — measures inter-rater reliability across judge personas.

WHAT THIS TESTS:
-----------------
The PSL system uses Groq llama-3.3-70b as an independent quality judge.
But LLM judges have a known weakness: they are sensitive to prompt wording.
A prompt saying "be demanding" vs "be supportive" can produce wildly different
scores for the SAME draft — making the metric untrustworthy.

This script quantifies that sensitivity using Cohen's weighted kappa:
  - We define 3 judge personas with different scoring biases
  - We score N drafts with all 3 judges
  - For each judge pair, we compute how often they agree (weighted kappa)

WHAT IS COHEN'S WEIGHTED KAPPA?
---------------------------------
Cohen's kappa (κ) measures agreement between two raters, corrected for
chance agreement.  The "weighted" variant (linear weights) also credits
near-misses: a judge giving 3 when another gives 4 is penalised less than
giving 1 when another gives 5.

  κ = (p_o - p_e) / (1 - p_e)

where p_o = observed agreement and p_e = expected agreement by chance.

Interpretation:
  κ > 0.80  → almost perfect (judge is reliable)
  κ 0.60–0.80 → substantial agreement
  κ 0.40–0.60 → moderate agreement
  κ < 0.40  → poor agreement (judge is biased by prompt wording)

THREE JUDGE PERSONAS:
----------------------
  strict   — Demands every claim have a citation. Penalises hard.
              Scores naturally land 1–3.
  balanced — Current production judge (neutral framing).
              Scores naturally land 2–4.
  lenient  — Focuses on overall coverage and intent.
              Minor missing citations tolerated.
              Scores naturally land 3–5.

A high kappa between strict and lenient means the judge is robust.
A low kappa means scores are prompt-dependent and less trustworthy.

Usage:
    python -m scripts.judge_tournament
    python -m scripts.judge_tournament --drafts 5 --url http://localhost:8000
    python -m scripts.judge_tournament --dry-run   (no server needed)
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL    = "http://localhost:8000"
TIMEOUT     = 120
N_DRAFTS    = 5    # number of drafts to generate and score
OUTPUT_PATH = Path("examples/outputs/judge_tournament.json")

QUERIES = [
    "Summarize the key compensation and payment terms",
    "Describe the termination and severance provisions",
    "What are the confidentiality and non-disclosure obligations?",
    "Summarize the intellectual property ownership clauses",
    "What are the governing law and arbitration terms?",
]

# ── Judge persona prompts ──────────────────────────────────────────────────────
# All three prompts ask for the same 4 dimensions on a 1-5 integer scale.
# The only difference is the framing and strictness level.

_JUDGE_STRICT = """\
You are a demanding senior legal partner reviewing a junior associate's draft.
You have zero tolerance for unverified claims. Penalise any sentence that lacks
an inline [E1]-style citation. Scores of 4 or 5 are reserved for flawless work.

Score using a 1-5 integer scale:
  1 = unacceptable  2 = below standard  3 = marginal  4 = meets standard  5 = exceptional

SCORING DIMENSIONS:
- groundedness:  EVERY factual claim must have a citation. Missing citations = score 1-2.
- completeness:  All key obligations and clauses must appear. Omissions = score 1-2.
- structure:     Clear headings, logical flow, professional tone required.
- overall:       Holistic quality. Reserved for truly professional output.

EVIDENCE:
{evidence_text}

DRAFT:
{draft_text}

Return ONLY valid JSON:
{{
  "groundedness": <1-5>,
  "completeness": <1-5>,
  "structure": <1-5>,
  "overall": <1-5>,
  "reasoning": "one sentence"
}}
"""

_JUDGE_BALANCED = """\
You are an independent legal document quality evaluator.

Score the following draft on each dimension using a 1-5 integer scale:
  1 = very poor  2 = poor  3 = acceptable  4 = good  5 = excellent

SCORING DIMENSIONS:
- groundedness:  Do all factual claims have inline [E1]-style citations?
- completeness:  Does the draft cover the key points from the evidence?
- structure:     Is the draft logically organised with clear sections?
- overall:       Holistic quality assessment.

EVIDENCE:
{evidence_text}

DRAFT:
{draft_text}

Return ONLY valid JSON:
{{
  "groundedness": <1-5>,
  "completeness": <1-5>,
  "structure": <1-5>,
  "overall": <1-5>,
  "reasoning": "one sentence"
}}
"""

_JUDGE_LENIENT = """\
You are a supportive legal writing coach reviewing a first-pass draft.
Focus on the drafter's intent and overall coverage rather than strict citation
mechanics. Minor missing citations are acceptable if the content is accurate.

Score using a 1-5 integer scale:
  1 = major gaps  2 = needs work  3 = good start  4 = solid draft  5 = excellent

SCORING DIMENSIONS:
- groundedness:  Does the overall content reflect the evidence?
                 Perfect citation mechanics not required for high scores.
- completeness:  Are the main topics addressed, even if briefly?
- structure:     Is it readable and reasonably professional?
- overall:       Would this be a useful starting point for further editing?

EVIDENCE:
{evidence_text}

DRAFT:
{draft_text}

Return ONLY valid JSON:
{{
  "groundedness": <1-5>,
  "completeness": <1-5>,
  "structure": <1-5>,
  "overall": <1-5>,
  "reasoning": "one sentence"
}}
"""

PERSONAS: dict[str, str] = {
    "strict":   _JUDGE_STRICT,
    "balanced": _JUDGE_BALANCED,
    "lenient":  _JUDGE_LENIENT,
}


# ── Cohen's weighted kappa ────────────────────────────────────────────────────

def _linear_weighted_kappa(
    ratings_a: list[int],
    ratings_b: list[int],
    min_r: int = 1,
    max_r: int = 5,
) -> float:
    """
    Compute linear weighted Cohen's kappa for two lists of integer ratings.

    Linear weights: w_ij = 1 - |i - j| / (max_r - min_r)
    This means a disagreement of 1 step (e.g., 3 vs 4) is penalised less
    than a full disagreement of 4 steps (e.g., 1 vs 5).

    Returns kappa in [-1, 1]. Values < 0 mean less agreement than chance.
    """
    n = len(ratings_a)
    if n == 0:
        return 0.0

    categories = list(range(min_r, max_r + 1))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Build weight matrix — linear weights
    weights = [
        [1.0 - abs(categories[i] - categories[j]) / (max_r - min_r)
         for j in range(k)]
        for i in range(k)
    ]

    # Build observed frequency matrix
    obs = [[0] * k for _ in range(k)]
    for ra, rb in zip(ratings_a, ratings_b):
        ia = cat_idx.get(ra, ra - min_r)
        ib = cat_idx.get(rb, rb - min_r)
        if 0 <= ia < k and 0 <= ib < k:
            obs[ia][ib] += 1

    # Marginal distributions
    row_totals = [sum(obs[i]) for i in range(k)]
    col_totals = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    # Observed weighted agreement
    p_o = sum(
        weights[i][j] * obs[i][j]
        for i in range(k) for j in range(k)
    ) / n

    # Expected weighted agreement (under independence assumption)
    p_e = sum(
        weights[i][j] * row_totals[i] * col_totals[j]
        for i in range(k) for j in range(k)
    ) / (n * n)

    if p_e >= 1.0:
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


# ── Groq judge call ───────────────────────────────────────────────────────────

def _call_groq_judge(
    groq_api_key: str,
    prompt_template: str,
    evidence_text: str,
    draft_text: str,
) -> dict:
    """Call Groq with the given judge prompt template and return scores."""
    from groq import Groq  # local import — only needed in live mode

    client = Groq(api_key=groq_api_key)
    prompt = prompt_template.format(
        evidence_text=evidence_text[:2000],
        draft_text=draft_text[:2000],
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ── Draft runner ──────────────────────────────────────────────────────────────

def _get_document_id(client: httpx.Client) -> str:
    resp = client.get("/documents")
    resp.raise_for_status()
    docs = resp.json().get("documents", [])
    if not docs:
        raise RuntimeError(
            "No documents found. Run `python -m scripts.seed` first."
        )
    return docs[0]["document_id"]


def _generate_draft(
    client: httpx.Client, document_id: str, query: str
) -> dict:
    resp = client.post(
        "/draft",
        json={"document_id": document_id, "query": query, "draft_type": "case_fact_summary"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_texts(draft_data: dict) -> tuple[str, str]:
    """Extract formatted evidence_text and draft_text from a /draft response."""
    evidence_parts = []
    draft_parts = []

    for section in draft_data.get("sections", []):
        title   = section.get("section_title", "")
        content = section.get("content", "")
        draft_parts.append(f"## {title}\n{content}")

        for ev_id, ev_data in section.get("evidence_map", {}).items():
            crumb   = ev_data.get("breadcrumb", "")
            ev_text = ev_data.get("content", "")[:400]
            evidence_parts.append(f"[{ev_id}] {crumb}\n{ev_text}")

    return "\n\n".join(evidence_parts[:6]), "\n\n".join(draft_parts)


# ── Dry-run simulation ─────────────────────────────────────────────────────────

def _run_synthetic(n: int) -> list[dict]:
    """
    Simulate judge scores without calling the server or Groq.

    Score distributions reflect typical behaviour of each persona:
      strict:   mean ≈ 2.5 (σ ≈ 0.6)
      balanced: mean ≈ 3.5 (σ ≈ 0.5)
      lenient:  mean ≈ 4.2 (σ ≈ 0.5)

    We use a seeded RNG so results are reproducible.
    """
    random.seed(42)

    def _score(mean: float, sigma: float) -> int:
        return max(1, min(5, round(random.gauss(mean, sigma))))

    records = []
    for i in range(n):
        query = QUERIES[i % len(QUERIES)]
        strict_scores   = {d: _score(2.5, 0.6) for d in ("groundedness","completeness","structure","overall")}
        balanced_scores = {d: _score(3.5, 0.5) for d in ("groundedness","completeness","structure","overall")}
        lenient_scores  = {d: _score(4.2, 0.5) for d in ("groundedness","completeness","structure","overall")}

        records.append({
            "draft_num": i + 1,
            "query":     query,
            "scores":    {
                "strict":   strict_scores,
                "balanced": balanced_scores,
                "lenient":  lenient_scores,
            },
            "synthetic": True,
        })

    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge tournament — compute inter-rater kappa across judge personas"
    )
    parser.add_argument("--drafts",  type=int, default=N_DRAFTS)
    parser.add_argument("--url",     default=BASE_URL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print("PSL Judge Tournament — Inter-rater Reliability (Cohen's κ)")
    print(f"{'='*65}")
    print(f"Drafts: {args.drafts} | Personas: {list(PERSONAS)} | Server: {args.url}")
    print(
        "\nSCORING: each draft scored 3× (strict / balanced / lenient judge).\n"
        "METRIC: linear weighted Cohen's kappa for each judge pair.\n"
        "TARGET: κ ≥ 0.60 (substantial agreement) between balanced↔strict/lenient.\n"
    )
    print(f"{'='*65}\n")

    # ── Collect scores ────────────────────────────────────────────────────────
    if args.dry_run:
        print("[DRY RUN] Generating synthetic scores (no server or Groq calls)...\n")
        records = _run_synthetic(args.drafts)
    else:
        # Load GROQ key from .env
        from python_service.config import settings
        if not settings.groq_api_key:
            print("ERROR: GROQ_API_KEY not set in .env — cannot run live judge tournament.")
            sys.exit(1)

        client      = httpx.Client(base_url=args.url, timeout=TIMEOUT)
        document_id = _get_document_id(client)
        print(f"Using document: {document_id}\n")

        records = []
        for i in range(args.drafts):
            query = QUERIES[i % len(QUERIES)]
            print(f"  Draft {i+1}/{args.drafts}: generating... ({query[:50]})")
            try:
                draft_data = _generate_draft(client, document_id, query)
            except Exception as exc:
                print(f"    Draft generation failed: {exc}")
                continue

            ev_text, draft_text = _extract_texts(draft_data)

            draft_scores: dict[str, dict] = {}
            for persona_name, prompt_tpl in PERSONAS.items():
                try:
                    scores = _call_groq_judge(
                        settings.groq_api_key, prompt_tpl, ev_text, draft_text
                    )
                    draft_scores[persona_name] = {
                        k: int(scores.get(k, 3))
                        for k in ("groundedness", "completeness", "structure", "overall")
                    }
                    print(
                        f"    {persona_name:<10} → "
                        f"G={draft_scores[persona_name]['groundedness']} "
                        f"C={draft_scores[persona_name]['completeness']} "
                        f"S={draft_scores[persona_name]['structure']} "
                        f"O={draft_scores[persona_name]['overall']}"
                    )
                    time.sleep(1)   # rate limit
                except Exception as exc:
                    print(f"    Judge {persona_name} failed: {exc}")
                    draft_scores[persona_name] = {"groundedness": 3, "completeness": 3, "structure": 3, "overall": 3}

            records.append({
                "draft_num": i + 1,
                "draft_id":  draft_data.get("draft_id"),
                "query":     query,
                "scores":    draft_scores,
            })
            time.sleep(2)

    if not records:
        print("No records collected — aborting.")
        return

    # ── Compute pairwise Cohen's kappa ────────────────────────────────────────
    persona_names = list(PERSONAS.keys())
    dimensions    = ["groundedness", "completeness", "structure", "overall"]

    kappa_results: dict[str, dict[str, float]] = {}

    for i, pa in enumerate(persona_names):
        for pb in persona_names[i+1:]:
            pair_key = f"{pa}_vs_{pb}"
            kappas: dict[str, float] = {}

            for dim in dimensions:
                ratings_a = [r["scores"][pa][dim] for r in records]
                ratings_b = [r["scores"][pb][dim] for r in records]
                kappas[dim] = round(_linear_weighted_kappa(ratings_a, ratings_b), 4)

            kappas["mean_kappa"] = round(sum(kappas.values()) / len(kappas), 4)
            kappa_results[pair_key] = kappas

    # ── Report ────────────────────────────────────────────────────────────────
    def _kappa_grade(k: float) -> str:
        if k >= 0.80: return "almost perfect"
        if k >= 0.60: return "substantial"
        if k >= 0.40: return "moderate"
        if k >= 0.20: return "fair"
        return "poor"

    print(f"\n{'='*65}")
    print("RESULTS — Cohen's Weighted κ by Judge Pair")
    print(f"{'='*65}")
    print(f"  {'Pair':<25} {'G':>5} {'C':>5} {'S':>5} {'O':>5} {'Mean':>6}  Grade")
    print(f"  {'-'*24} {'---':>5} {'---':>5} {'---':>5} {'---':>5} {'----':>6}")
    for pair, kappas in kappa_results.items():
        g = kappas.get("groundedness", 0)
        c = kappas.get("completeness", 0)
        s = kappas.get("structure", 0)
        o = kappas.get("overall", 0)
        m = kappas.get("mean_kappa", 0)
        grade = _kappa_grade(m)
        print(f"  {pair:<25} {g:>5.3f} {c:>5.3f} {s:>5.3f} {o:>5.3f} {m:>6.3f}  {grade}")

    print(f"\n  Score means by persona:")
    for name in persona_names:
        for dim in dimensions:
            vals = [r["scores"][name][dim] for r in records]
            mean = sum(vals) / len(vals) if vals else 0
            print(f"    {name:<10} {dim:<15} → {mean:.2f}")
    print(f"{'='*65}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "n_drafts":       len(records),
        "personas":       list(PERSONAS.keys()),
        "dimensions":     dimensions,
        "kappa_results":  kappa_results,
        "interpretation": {
            k: f"mean κ={v['mean_kappa']:.3f} — {_kappa_grade(v['mean_kappa'])}"
            for k, v in kappa_results.items()
        },
        "target_kappa":   0.60,
        "records":        records,
        "dry_run":        args.dry_run,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Results saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
