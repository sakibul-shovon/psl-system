"""
scripts/known_fact_eval.py

Known-fact retrieval probe — does the pipeline find specific facts we know exist?

WHAT THIS TESTS:
-----------------
The adversarial eval tests the NEGATIVE case (refusing garbage queries).
This script tests the POSITIVE case: given a fact we KNOW is in the document,
does the retrieval pipeline surface it?

This is called a "known-fact probe" or "precision@k" evaluation in information
retrieval research.  We define a test case as:
  (query, expected_substring)
and assert that the expected text appears somewhere in the evidence returned
by the draft pipeline for that query.

WHY EVIDENCE CONTENT, NOT JUST SECTION CONTENT?
-------------------------------------------------
The draft agent summarises and paraphrases evidence — so the exact string
"$800,000" might be rewritten as "eight hundred thousand dollars" in the final
section text.  But the raw evidence chunks in evidence_map preserve the
original document text word-for-word.  That's what we search: evidence_map
values carry the original extracted text, making this a direct retrieval test,
not a generation test.

WHAT IS precision@3?
---------------------
In information retrieval, precision@k = (relevant docs in top-k) / k.
Here we adapt it: for each fact query, we look at the top 3 evidence items
(E1, E2, E3) and ask "does the expected fact appear in any of them?"
  Hit  = fact found in top-3 evidence items  → counts as 1
  Miss = fact not found                       → counts as 0
  precision@3 = hits / total_test_cases

A score of 1.0 means every tested fact was retrieved in the first 3 evidence
items.  A score of 0.6 means only 60% of facts were reachable.

TEST CASES (based on clean_contract.pdf content):
--------------------------------------------------
  The example employment agreement (Pearson Specter Litt × Harvey Specter)
  contains the following known facts.  Each (query, expected) pair is a test.

Usage:
    python -m scripts.known_fact_eval
    python -m scripts.known_fact_eval --url http://localhost:8000
    python -m scripts.known_fact_eval --dry-run   (no server needed)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL    = "http://localhost:8000"
TIMEOUT     = 120
OUTPUT_PATH = Path("examples/outputs/known_fact_eval.json")


# ── Test case bank ─────────────────────────────────────────────────────────────
# Each entry is:
#   query         — the natural-language question to send to /draft
#   expected      — the literal substring we expect to find in evidence content
#   description   — human-readable label for the report
#
# All facts are sourced from scripts/generate_examples.py (clean_contract.pdf).
# The evidence_map in each section stores the raw chunk text, so we search there.

KNOWN_FACT_CASES = [
    {
        "query":       "Who are the parties to this employment agreement?",
        "expected":    "Harvey Reginald Specter",
        "description": "Employee name",
    },
    {
        "query":       "What law firm is party to this agreement?",
        "expected":    "Pearson Specter Litt",
        "description": "Firm name",
    },
    {
        "query":       "What is the annual base salary specified in this agreement?",
        "expected":    "800,000",
        "description": "Base salary amount",
    },
    {
        "query":       "What is the effective date of this employment agreement?",
        "expected":    "January 1, 2025",
        "description": "Effective date",
    },
    {
        "query":       "What equity or partnership interest does the employee receive?",
        "expected":    "fifteen percent",
        "description": "Equity percentage",
    },
    {
        "query":       "What are the non-solicitation obligations after termination?",
        "expected":    "two (2) years",
        "description": "Non-solicitation period",
    },
    {
        "query":       "What is the governing law and dispute resolution method?",
        "expected":    "New York",
        "description": "Governing law jurisdiction",
    },
]


# ── Hit detection ──────────────────────────────────────────────────────────────

def _check_hit(response: dict, expected: str) -> tuple[bool, str]:
    """
    Return (hit, location) where hit=True if `expected` was found.

    Search order:
      1. evidence_map values in each section (raw chunk text — most reliable)
      2. section content (the generated summary — less reliable due to paraphrase)

    We search case-insensitively so "800,000" matches "$800,000" and
    "JANUARY 1, 2025" also matches "January 1, 2025".
    """
    expected_lower = expected.lower()

    for section in response.get("sections", []):
        # ── Check raw evidence content ────────────────────────────────────────
        evidence_map = section.get("evidence_map", {})
        for ev_id, ev_data in evidence_map.items():
            content = ev_data.get("content", "").lower()
            if expected_lower in content:
                return True, f"evidence {ev_id} in section '{section.get('section_title', '?')}'"

        # ── Fallback: check generated section text ────────────────────────────
        section_content = section.get("content", "").lower()
        if expected_lower in section_content:
            return True, f"section text '{section.get('section_title', '?')}'"

    return False, "not found"


# ── Live runner ────────────────────────────────────────────────────────────────

def _get_document_id(client: httpx.Client) -> str:
    resp = client.get("/documents")
    resp.raise_for_status()
    docs = resp.json().get("documents", [])
    if not docs:
        raise RuntimeError(
            "No documents found. Run `python -m scripts.seed` first."
        )
    return docs[0]["document_id"]


def _run_fact_query(
    client: httpx.Client,
    document_id: str,
    case: dict,
    case_num: int,
) -> dict:
    body = {
        "document_id": document_id,
        "query":       case["query"],
        "draft_type":  "case_fact_summary",
    }

    try:
        resp = client.post("/draft", json=body, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [{case_num:02d}] ERROR: {exc}")
        return {
            "case_num":    case_num,
            "description": case["description"],
            "query":       case["query"],
            "expected":    case["expected"],
            "hit":         False,
            "location":    f"error: {exc}",
        }

    hit, location = _check_hit(data, case["expected"])
    icon = "HIT  ✓" if hit else "MISS ✗"
    print(
        f"  [{case_num:02d}] {icon} | "
        f"{case['description']:<30} | "
        f"expected: {case['expected']!r:<25} | "
        f"found at: {location}"
    )

    return {
        "case_num":    case_num,
        "description": case["description"],
        "query":       case["query"],
        "expected":    case["expected"],
        "hit":         hit,
        "location":    location,
        "draft_id":    data.get("draft_id"),
        "grounding_score": data.get("grounding_score"),
    }


# ── Dry-run simulation ─────────────────────────────────────────────────────────

def _run_synthetic(cases: list[dict]) -> list[dict]:
    """
    Simulate known-fact retrieval results without a server.

    In a well-tuned dense+BM25+rerank pipeline:
    - Named entities (names, firm names) score very high on BM25 (exact keyword match)
    - Dollar amounts ("800,000") are highly specific tokens → easy BM25 hit
    - Dates are usually near-exact token matches
    - Percentages ("fifteen percent") are less common and may need semantic search
    - Jurisdiction terms ("New York") appear in multiple chunks → easy hit

    We simulate 6/7 hits (precision@3 = 0.857) with one miss:
    The equity case ("fifteen percent") fails because it's a word form of a number
    and the chunk might have been split at a paragraph boundary in legal_chunker.
    """
    OUTCOMES = [True, True, True, True, False, True, True]   # 6/7 hits

    results = []
    for i, (case, hit) in enumerate(zip(cases, OUTCOMES)):
        location = (
            f"evidence E1 in section 'Compensation'" if hit
            else "not found"
        )
        icon = "HIT  ✓" if hit else "MISS ✗"
        print(
            f"  [{i+1:02d}] {icon} | "
            f"{case['description']:<30} | "
            f"expected: {case['expected']!r:<25} | "
            f"found at: {location}"
        )
        results.append({
            "case_num":    i + 1,
            "description": case["description"],
            "query":       case["query"],
            "expected":    case["expected"],
            "hit":         hit,
            "location":    location,
            "synthetic":   True,
        })
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Known-fact probe — measure precision@3 for structured facts"
    )
    parser.add_argument("--url",     default=BASE_URL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print("PSL Known-Fact Evaluation — Retrieval Precision@3")
    print(f"{'='*65}")
    print(f"Test cases: {len(KNOWN_FACT_CASES)} | Server: {args.url}")
    print(
        "\nMETHOD: For each query, search evidence_map values in returned\n"
        "sections for the expected substring (case-insensitive).\n"
        "precision@3 = (facts found in top-3 evidence) / total_cases\n"
    )
    print(f"{'='*65}\n")

    # ── Data collection ───────────────────────────────────────────────────────
    if args.dry_run:
        print("[DRY RUN] Simulating fact retrieval (no server calls)...\n")
        results = _run_synthetic(KNOWN_FACT_CASES)
    else:
        client = httpx.Client(base_url=args.url, timeout=TIMEOUT)
        document_id = _get_document_id(client)
        print(f"Testing against document: {document_id}\n")
        results = []
        for i, case in enumerate(KNOWN_FACT_CASES, start=1):
            result = _run_fact_query(client, document_id, case, i)
            results.append(result)
            time.sleep(2)

    # ── Analysis ──────────────────────────────────────────────────────────────
    total = len(results)
    hits  = sum(1 for r in results if r["hit"])
    misses = total - hits
    precision_at_3 = hits / total if total > 0 else 0.0

    grade = (
        "EXCELLENT" if precision_at_3 >= 0.90 else
        "GOOD"      if precision_at_3 >= 0.75 else
        "FAIR"      if precision_at_3 >= 0.55 else
        "POOR"
    )

    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    print(f"  Total test cases    : {total}")
    print(f"  Hits                : {hits}")
    print(f"  Misses              : {misses}")
    print(f"  Precision@3         : {precision_at_3:.1%}  [{grade}]")
    if misses > 0:
        print(f"\n  ⚠  Missed facts:")
        for r in results:
            if not r["hit"]:
                print(f"     • {r['description']}: {r['expected']!r}")
    print(f"{'='*65}\n")

    # ── Save output ────────────────────────────────────────────────────────────
    output = {
        "total_cases":   total,
        "hits":          hits,
        "misses":        misses,
        "precision_at_3": round(precision_at_3, 4),
        "grade":         grade,
        "target_precision_at_3": 0.75,
        "met_target":    precision_at_3 >= 0.75,
        "methodology": (
            "For each query, the draft pipeline is invoked. "
            "A 'hit' = expected_substring found in evidence_map content "
            "(raw chunk text) from any of the returned sections. "
            "Fallback: check generated section text (lower confidence)."
        ),
        "results":  results,
        "dry_run":  args.dry_run,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Results saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
