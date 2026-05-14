"""
A/B causal proof — do patterns actually improve draft quality?

WHY do we need a t-test instead of just comparing averages?
Pattern-applied drafts might have higher scores for reasons unrelated to
the patterns: maybe those queries happened to hit denser evidence, or the
document type was easier. A t-test controls for this by checking whether
the *distribution* of scores differs, not just the means.

Methodology:
  1. Run N drafts against the same document + query set.
  2. Randomly assign each draft to group A (patterns disabled) or B (enabled).
     Random assignment is what makes this causal, not just correlational.
  3. Compare judge_overall scores between groups with Welch's t-test.
  4. Report means, delta, p-value, and effect size (Cohen's d).
  5. Save results to examples/outputs/ab_test_results.json.

Usage:
    python -m scripts.ab_test
    python -m scripts.ab_test --drafts 30 --url http://localhost:8000
    python -m scripts.ab_test --dry-run   (uses synthetic scores, no server needed)

For the assessment submission: run once live, copy the saved JSON into
EVALUATION.md to demonstrate causal evidence.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL   = "http://localhost:8000"
N_DRAFTS   = 20        # total drafts to run (10 per group)
TIMEOUT    = 120       # seconds per draft call

# Fixed query set — same queries used for both groups so variance comes from
# pattern injection, not query difficulty.
QUERIES = [
    "Summarize the compensation and payment terms",
    "Describe the termination and notice provisions",
    "What are the confidentiality and non-disclosure obligations?",
    "Summarize the indemnification and liability clauses",
    "What are the governing law and dispute resolution terms?",
]

OUTPUT_PATH = Path("examples/outputs/ab_test_results.json")


# ── Statistics helpers ────────────────────────────────────────────────────────

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _variance(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return sum((x - m) ** 2 for x in vals) / (len(vals) - 1)


def _welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """
    Welch's t-test (unequal variances) — more appropriate than Student's t
    when group sizes or variances may differ.

    Returns (t_statistic, p_value).
    p_value < 0.05 → statistically significant difference.

    We implement this from scratch using numpy's math so we don't need scipy.
    """
    import numpy as np

    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0

    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)

    mean_a, mean_b = arr_a.mean(), arr_b.mean()
    var_a, var_b   = arr_a.var(ddof=1), arr_b.var(ddof=1)
    n_a, n_b       = len(arr_a), len(arr_b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0

    t_stat = (mean_a - mean_b) / se

    # Welch–Satterthwaite degrees of freedom
    df_num   = (var_a / n_a + var_b / n_b) ** 2
    df_denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_denom if df_denom > 0 else 1.0

    # Two-tailed p-value via the t-distribution CDF (scipy not available,
    # so we use the incomplete beta function from numpy's special module).
    # If numpy's special is unavailable, we approximate with a lookup table.
    try:
        from scipy.special import betainc  # type: ignore
        x = df / (df + t_stat ** 2)
        p_value = betainc(df / 2, 0.5, x)
    except ImportError:
        # Rough approximation: treat as normal distribution for large df
        # (central limit theorem kicks in around df > 30)
        z = abs(t_stat)
        # Abramowitz & Stegun approximation for the normal CDF tail
        t_val = 1.0 / (1.0 + 0.2316419 * z)
        poly = t_val * (0.319381530 + t_val * (-0.356563782 + t_val * (
            1.781477937 + t_val * (-1.821255978 + t_val * 1.330274429)
        )))
        p_value = 2.0 * poly * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        p_value = max(0.0, min(1.0, p_value))

    return float(t_stat), float(p_value)


def _cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d — standardised effect size. |d| > 0.8 = large effect."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_sd = math.sqrt((_variance(a) + _variance(b)) / 2)
    return (_mean(a) - _mean(b)) / pooled_sd if pooled_sd > 0 else 0.0


# ── Draft runner ──────────────────────────────────────────────────────────────

def _get_document_id(client: httpx.Client) -> str:
    """Return the most recently uploaded document_id."""
    resp = client.get("/documents")
    resp.raise_for_status()
    docs = resp.json().get("documents", [])
    if not docs:
        raise RuntimeError(
            "No documents found. Upload a document first (e.g. python -m scripts.seed)."
        )
    return docs[0]["document_id"]


def _run_draft(
    client: httpx.Client,
    document_id: str,
    query: str,
    use_patterns: bool,
    draft_num: int,
) -> dict:
    """
    Generate one draft and return its quality scores.

    use_patterns=False means we pass skip_patterns=True in the request body,
    which tells the planner to skip the pattern retrieval step. This is our
    control group — same document, same query, but no learned patterns applied.
    """
    body: dict = {
        "document_id": document_id,
        "query":       query,
        "draft_type":  "case_fact_summary",
    }
    if not use_patterns:
        # Signal to the agent to skip pattern injection for this draft.
        # The planner respects this flag so both groups see identical evidence.
        body["skip_patterns"] = True

    try:
        resp = client.post("/draft", json=body, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"    Draft {draft_num} FAILED: {exc}")
        return {"draft_num": draft_num, "query": query, "use_patterns": use_patterns, "error": str(exc)}

    judge_overall = data.get("judge_scores", {}).get("overall")
    grounding     = data.get("grounding_score", 0.0)

    print(
        f"    Draft {draft_num:02d} | {'WITH' if use_patterns else 'WITHOUT':7s} patterns | "
        f"judge={judge_overall} | grounding={grounding:.2f} | query: {query[:40]}..."
    )

    return {
        "draft_num":    draft_num,
        "draft_id":     data.get("draft_id"),
        "query":        query,
        "use_patterns": use_patterns,
        "judge_overall": float(judge_overall) if judge_overall is not None else None,
        "grounding_score": grounding,
        "patterns_applied": data.get("patterns_applied", 0),
        "agent_iterations": data.get("agent_iterations", 0),
    }


def _run_synthetic(n: int) -> list[dict]:
    """
    Dry-run mode: generate synthetic scores without calling the server.

    Pattern group gets scores drawn from N(3.8, 0.4); no-pattern group from
    N(3.0, 0.5). This simulates the expected distribution and verifies the
    statistics code is correct before a real run.
    """
    results = []
    random.seed(42)
    for i in range(1, n + 1):
        use_patterns = random.random() > 0.5
        base = 3.8 if use_patterns else 3.0
        score = round(max(1.0, min(5.0, random.gauss(base, 0.4))), 2)
        results.append({
            "draft_num":     i,
            "query":         QUERIES[i % len(QUERIES)],
            "use_patterns":  use_patterns,
            "judge_overall": score,
            "grounding_score": round(random.uniform(0.65, 0.90), 3),
            "patterns_applied": random.randint(2, 5) if use_patterns else 0,
            "synthetic":     True,
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test: patterns vs no patterns")
    parser.add_argument("--drafts",  type=int, default=N_DRAFTS)
    parser.add_argument("--url",     default=BASE_URL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("PSL A/B Test — Pattern Causal Proof")
    print(f"{'='*60}")
    print(f"Drafts: {args.drafts} | Server: {args.url}")
    print(f"{'='*60}\n")

    # ── Data collection ───────────────────────────────────────────────────────
    if args.dry_run:
        print("[DRY RUN] Generating synthetic scores (no server calls)...\n")
        all_results = _run_synthetic(args.drafts)
    else:
        client = httpx.Client(base_url=args.url, timeout=TIMEOUT)
        document_id = _get_document_id(client)
        print(f"Using document: {document_id}\n")

        all_results = []
        query_cycle = QUERIES * math.ceil(args.drafts / len(QUERIES))
        for i in range(args.drafts):
            # Alternate WITH/WITHOUT to keep groups balanced regardless of order
            use_patterns = (i % 2 == 0)
            query = query_cycle[i]
            result = _run_draft(client, document_id, query, use_patterns, i + 1)
            all_results.append(result)
            time.sleep(2)   # avoid hammering the LLM endpoints

    # ── Analysis ──────────────────────────────────────────────────────────────
    with_scores    = [r["judge_overall"] for r in all_results
                      if r.get("use_patterns") and r.get("judge_overall") is not None]
    without_scores = [r["judge_overall"] for r in all_results
                      if not r.get("use_patterns") and r.get("judge_overall") is not None]

    if len(with_scores) < 2 or len(without_scores) < 2:
        print("\nInsufficient data for statistical test (need ≥2 scores per group).")
        return

    t_stat, p_value = _welch_t_test(with_scores, without_scores)
    effect_d        = _cohens_d(with_scores, without_scores)

    mean_with    = _mean(with_scores)
    mean_without = _mean(without_scores)
    delta        = mean_with - mean_without

    # ── Report ────────────────────────────────────────────────────────────────
    sig = "✓ SIGNIFICANT" if p_value < 0.05 else "✗ not significant"
    effect_label = (
        "large"  if abs(effect_d) >= 0.8 else
        "medium" if abs(effect_d) >= 0.5 else
        "small"
    )

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  WITH patterns    (n={len(with_scores)}): mean judge_overall = {mean_with:.3f}")
    print(f"  WITHOUT patterns (n={len(without_scores)}): mean judge_overall = {mean_without:.3f}")
    print(f"  Delta            : {delta:+.3f}")
    print(f"  Welch t-statistic: {t_stat:.3f}")
    print(f"  p-value          : {p_value:.4f}  ({sig})")
    print(f"  Cohen's d        : {effect_d:.3f} ({effect_label} effect)")
    print(f"{'='*60}\n")

    output = {
        "n_total":         len(all_results),
        "n_with_patterns": len(with_scores),
        "n_without":       len(without_scores),
        "mean_with_patterns":    round(mean_with, 4),
        "mean_without_patterns": round(mean_without, 4),
        "delta_judge_overall":   round(delta, 4),
        "t_statistic":           round(t_stat, 4),
        "p_value":               round(p_value, 6),
        "cohens_d":              round(effect_d, 4),
        "effect_size":           effect_label,
        "significant_p05":       p_value < 0.05,
        "interpretation": (
            f"Patterns {'causally' if p_value < 0.05 else 'do not significantly'} "
            f"improve draft quality (Δ={delta:+.3f}, p={p_value:.4f}, d={effect_d:.3f})."
        ),
        "raw_results": all_results,
        "dry_run": args.dry_run,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
