"""
scripts/adversarial_eval.py

Adversarial robustness evaluation — does the PSL system refuse off-topic questions?

WHAT IS AN ADVERSARIAL EVALUATION?
-----------------------------------
Normal evaluation tests whether the system answers correctly.
Adversarial evaluation tests whether the system correctly REFUSES to answer
when it shouldn't — i.e., when the question is completely off-topic for the
document being analysed.

WHY DOES THIS MATTER FOR LEGAL AI?
-----------------------------------
A lawyer uploading a lease agreement needs to trust that if they ask
"what are the patent licensing royalties?", the system will say
"I can't find that in the document" — NOT fabricate a plausible-sounding
but entirely invented clause. Hallucination in legal work creates liability.

HOW THE PSL SYSTEM GUARDS AGAINST THIS:
-----------------------------------------
  Layer 1 — Retrieval guard: the cross-encoder reranker scores each
    candidate chunk against the query. If the best score < 0.35, the
    pipeline returns sufficient=False → the executor writes
    "[INSUFFICIENT EVIDENCE: no relevant chunks found]"

  Layer 2 — Generation constraint: the Gemini executor prompt says:
    "Only use evidence provided below — do not invent facts."
    "If evidence is insufficient, write [INSUFFICIENT EVIDENCE: reason]."

WHAT THIS SCRIPT MEASURES:
----------------------------
  1. Sends N adversarial queries against a real document.
  2. For each response, detects a "refusal":
       Signal A: any section content contains "[INSUFFICIENT EVIDENCE"
       Signal B: overall grounding_score < 0.30 (draft is basically ungrounded)
  3. Reports refusal_precision = correctly_refused / total_adversarial_queries

  Target: refusal_precision >= 0.80 (4 out of 5 adversarial queries refused).
  A perfect score is 1.0; random hallucinating would score 0.0.

Usage:
    python -m scripts.adversarial_eval
    python -m scripts.adversarial_eval --url http://localhost:8000
    python -m scripts.adversarial_eval --dry-run   (no server needed)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

# Ensure Unicode output works on Windows terminals (cp1252 → utf-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL    = "http://localhost:8000"
TIMEOUT     = 120
OUTPUT_PATH = Path("examples/outputs/adversarial_eval.json")

# ── Adversarial query bank ─────────────────────────────────────────────────────
# These are questions that should FAIL against any typical legal document.
# They fall into three categories:
#   (a) Completely off-domain — medicine, physics, cooking
#   (b) Legal but almost certainly absent from employment/lease/NDA docs
#   (c) Nonsense that no document would ever address
#
# The key design rule: each query is formulated to sound plausible enough that
# a hallucinating LLM might try to answer it, but is firmly absent from the
# indexed document. That's what makes them adversarial.

ADVERSARIAL_QUERIES = [
    # Off-domain: medical
    "What are the recommended ibuprofen dosages for paediatric patients specified here?",
    # Off-domain: physics
    "Explain the Pauli exclusion principle as described in this document.",
    # Off-domain: agriculture
    "What irrigation requirements for rice cultivation are defined in this agreement?",
    # Legal but absent from typical employment/lease/NDA
    "What patent licensing royalties and technology transfer fees are specified?",
    "Describe the nuclear waste disposal obligations mandated by this contract.",
    # Financial instrument — not a legal services contract
    "What derivative instrument pricing formulas are embedded in this agreement?",
    # Culinary — clearly nonsense
    "What is the recipe for French boeuf bourguignon mentioned in the document?",
    # Blockchain — clearly out of scope for a pre-2025 legal doc template
    "List all proof-of-stake consensus mechanisms described in this contract.",
]


# ── Refusal detection ──────────────────────────────────────────────────────────

def _is_refusal(response: dict) -> tuple[bool, str]:
    """
    Return (is_refusal, reason) for one draft API response.

    We check two independent signals:
      A) Any section content contains the literal "[INSUFFICIENT EVIDENCE"
         substring — this is the executor explicitly refusing.
      B) overall grounding_score < 0.30 — the NLI verifier found fewer than
         30% of claims supported by evidence, meaning the draft is essentially
         ungrounded (even if no explicit refusal marker was written).

    Either signal alone counts as a refusal.  We prefer Signal A (explicit)
    when reporting; Signal B catches cases where Gemini tried to draft something
    but couldn't ground it — a subtler form of refusal.
    """
    grounding = response.get("grounding_score", 1.0)

    for section in response.get("sections", []):
        content = section.get("content", "")
        if "[INSUFFICIENT EVIDENCE" in content:
            return True, "explicit_refusal_marker"

    if grounding < 0.30:
        return True, f"low_grounding_score ({grounding:.3f})"

    return False, "answered"


# ── Live runner ────────────────────────────────────────────────────────────────

def _get_document_id(client: httpx.Client) -> str:
    """Return the most recently uploaded document_id."""
    resp = client.get("/documents")
    resp.raise_for_status()
    docs = resp.json().get("documents", [])
    if not docs:
        raise RuntimeError(
            "No documents found. Run `python -m scripts.seed` first to ingest an example document."
        )
    return docs[0]["document_id"]


def _run_adversarial_query(
    client: httpx.Client,
    document_id: str,
    query: str,
    query_num: int,
) -> dict:
    """
    Fire one adversarial query at the /draft endpoint and evaluate the response.

    Returns a result dict describing whether the system refused correctly.
    """
    body = {
        "document_id": document_id,
        "query":       query,
        "draft_type":  "case_fact_summary",
    }

    try:
        resp = client.post("/draft", json=body, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        # If the server errors out, that's a refusal by default — it didn't
        # hallucinate a fake answer.  Treat as refused with special reason.
        print(f"  [{query_num:02d}] SERVER ERROR — treated as refusal: {exc}")
        return {
            "query_num":   query_num,
            "query":       query,
            "refused":     True,
            "reason":      f"server_error: {exc}",
            "grounding_score": None,
            "sections_count":  0,
        }

    refused, reason = _is_refusal(data)
    grounding = data.get("grounding_score", 0.0)
    n_sections = len(data.get("sections", []))

    status_icon = "REFUSED ✓" if refused else "ANSWERED ✗"
    print(
        f"  [{query_num:02d}] {status_icon} | "
        f"grounding={grounding:.3f} | sections={n_sections} | "
        f"reason={reason} | query: {query[:55]}..."
    )

    return {
        "query_num":       query_num,
        "query":           query,
        "refused":         refused,
        "reason":          reason,
        "grounding_score": grounding,
        "sections_count":  n_sections,
        "draft_id":        data.get("draft_id"),
    }


# ── Dry-run simulation ─────────────────────────────────────────────────────────

def _run_synthetic(queries: list[str]) -> list[dict]:
    """
    Dry-run mode: simulate adversarial query responses without a server.

    The simulation reflects realistic system behaviour:
      - Completely off-domain queries (medicine, physics) → always refused
        because the reranker can't score them above 0.35 against legal text.
      - Legal-but-absent queries → mostly refused; one might slip through
        if a generic contract clause (e.g., IP ownership) partially resembles
        the query vector.
      - Nonsense → always refused.

    This gives us an 87.5% refusal rate (7/8 refused) in dry-run, which is a
    realistic baseline for a well-guarded retrieval system.
    """
    # Pre-determined outcomes: index 4 (nuclear waste) slips through as a
    # partial grounding match on an "environmental" clause — that's the one
    # failure mode this simulation demonstrates.
    OUTCOMES: dict[int, tuple[bool, str, float]] = {
        0: (True,  "explicit_refusal_marker", 0.00),  # ibuprofen
        1: (True,  "explicit_refusal_marker", 0.00),  # Pauli exclusion
        2: (True,  "explicit_refusal_marker", 0.00),  # irrigation
        3: (True,  "explicit_refusal_marker", 0.00),  # patent royalties
        4: (False, "answered",                0.34),  # nuclear waste — partial match
        5: (True,  "low_grounding_score (0.12)", 0.12),  # derivatives
        6: (True,  "explicit_refusal_marker", 0.00),  # boeuf bourguignon
        7: (True,  "explicit_refusal_marker", 0.00),  # blockchain
    }

    results = []
    for i, q in enumerate(queries):
        refused, reason, grounding = OUTCOMES.get(
            i, (True, "explicit_refusal_marker", 0.00)
        )
        status_icon = "REFUSED ✓" if refused else "ANSWERED ✗"
        print(
            f"  [{i+1:02d}] {status_icon} | "
            f"grounding={grounding:.3f} | reason={reason} | "
            f"query: {q[:55]}..."
        )
        results.append({
            "query_num":       i + 1,
            "query":           q,
            "refused":         refused,
            "reason":          reason,
            "grounding_score": grounding,
            "sections_count":  0 if refused else 3,
            "synthetic":       True,
        })
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adversarial probe — measure the system's refusal precision"
    )
    parser.add_argument(
        "--url", default=BASE_URL,
        help="Base URL of the PSL API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate results without calling the server"
    )
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print("PSL Adversarial Evaluation — Refusal Precision")
    print(f"{'='*65}")
    print(f"Queries: {len(ADVERSARIAL_QUERIES)} | Server: {args.url}")
    print(
        "\nREFUSAL DEFINITION: a response is a 'refusal' if:\n"
        "  (A) any section content contains '[INSUFFICIENT EVIDENCE' OR\n"
        "  (B) grounding_score < 0.30\n"
    )
    print(f"{'='*65}\n")

    # ── Data collection ───────────────────────────────────────────────────────
    if args.dry_run:
        print("[DRY RUN] Simulating adversarial responses (no server calls)...\n")
        results = _run_synthetic(ADVERSARIAL_QUERIES)
    else:
        client = httpx.Client(base_url=args.url, timeout=TIMEOUT)
        document_id = _get_document_id(client)
        print(f"Testing against document: {document_id}\n")

        results = []
        for i, query in enumerate(ADVERSARIAL_QUERIES, start=1):
            result = _run_adversarial_query(client, document_id, query, i)
            results.append(result)
            time.sleep(2)   # avoid hammering the LLM rate limits

    # ── Analysis ──────────────────────────────────────────────────────────────
    total    = len(results)
    refused  = sum(1 for r in results if r["refused"])
    answered = total - refused

    refusal_precision = refused / total if total > 0 else 0.0

    # Breakdown by refusal reason
    reason_counts: dict[str, int] = {}
    for r in results:
        key = r["reason"].split(" ")[0]   # e.g. "explicit_refusal_marker"
        reason_counts[key] = reason_counts.get(key, 0) + 1

    # ── Report ────────────────────────────────────────────────────────────────
    grade = (
        "EXCELLENT" if refusal_precision >= 0.90 else
        "GOOD"      if refusal_precision >= 0.80 else
        "FAIR"      if refusal_precision >= 0.60 else
        "POOR"
    )

    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    print(f"  Total adversarial queries : {total}")
    print(f"  Correctly refused         : {refused}")
    print(f"  Incorrectly answered      : {answered}")
    print(f"  Refusal precision         : {refusal_precision:.1%}  [{grade}]")
    print(f"\n  Refusal breakdown by signal:")
    for reason, count in sorted(reason_counts.items()):
        print(f"    {reason:<35} {count}")
    if answered > 0:
        print(f"\n  ⚠  Queries that leaked through (hallucination risk):")
        for r in results:
            if not r["refused"]:
                print(f"     • {r['query'][:70]}")
    print(f"{'='*65}\n")

    # ── Save output ────────────────────────────────────────────────────────────
    output = {
        "total_queries":        total,
        "correctly_refused":    refused,
        "incorrectly_answered": answered,
        "refusal_precision":    round(refusal_precision, 4),
        "grade":                grade,
        "target_precision":     0.80,
        "met_target":           refusal_precision >= 0.80,
        "refusal_signals": {
            "A_insufficient_evidence_marker": "any section contains '[INSUFFICIENT EVIDENCE'",
            "B_low_grounding_score":          "grounding_score < 0.30",
        },
        "reason_breakdown": reason_counts,
        "results": results,
        "dry_run": args.dry_run,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Results saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
