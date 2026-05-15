"""
Seed script — pre-loads realistic operator edits so the system has learned
patterns from day one of a demo.

Run once after initial setup:
    python -m scripts.seed

What it does:
  1. Checks whether any documents exist in SQLite.
     If NOT → auto-ingests examples/inputs/clean_contract.pdf (bootstrap mode).
  2. Waits for the ingestion pipeline to finish (polls GET /job/{id}).
  3. Generates a BASELINE draft (no patterns yet) and saves it to
     examples/outputs/draft_baseline.json.
  4. POSTs several realistic lawyer edits via /feedback.
  5. Waits for background pattern extraction to finish.
  6. Generates an IMPROVED draft (with the freshly learned patterns) and saves
     it to examples/outputs/draft_improved.json.
  7. Prints a summary of extracted patterns.

Why seed instead of using real edits?
  The improvement-report endpoint needs drafts both before AND after patterns
  are applied to compute a delta.  Without seeding, a fresh install shows
  "not enough data."  Seeding gives the demo a meaningful baseline.

Why auto-ingest?
  A reviewer who just cloned the repo should be able to run `python -m scripts.seed`
  once (with the API server running) and get a fully working demo — no manual
  PDF upload step required.
"""

import json
import time
from pathlib import Path

import httpx
from sqlmodel import Session, select

BASE_URL  = "http://localhost:8000"
EXAMPLES  = Path(__file__).parent.parent / "examples"
SEED_PDF  = EXAMPLES / "inputs" / "clean_contract.pdf"
OUTPUTS   = EXAMPLES / "outputs"

# Realistic before/after edit pairs that cover different pattern types.
# These are the "operator ideal" texts the edit_distance_trend script also uses
# as ground truth, so the two evaluation tools stay consistent.
SEED_EDITS = [
    {
        "section_id": "seed_1",
        "section_title": "Compensation",
        "original_text": (
            "The employee will receive a salary of an amount to be determined by the board. "
            "This will be paid monthly."
        ),
        "edited_text": (
            "Employee shall receive Base Compensation in an amount to be determined by the Board of Directors. "
            "Such Base Compensation shall be payable in equal monthly installments [E1]."
        ),
    },
    {
        "section_id": "seed_2",
        "section_title": "Termination",
        "original_text": (
            "The company can fire the employee for any reason without notice."
        ),
        "edited_text": (
            "The Company may terminate Employee's employment at will upon delivery of written notice "
            "in accordance with Section 11 hereof [E1]."
        ),
    },
    {
        "section_id": "seed_3",
        "section_title": "Severance",
        "original_text": (
            "If fired without cause, the employee gets 3x their yearly pay as a one-time payment."
        ),
        "edited_text": (
            "Upon termination without cause, Employee shall receive a lump sum equal to three (3) times "
            "Employee's Base Compensation, payable within fifteen (15) days of the Date of Termination [E1]."
        ),
    },
    {
        "section_id": "seed_4",
        "section_title": "Benefits",
        "original_text": (
            "Health benefits continue for 3 years after termination."
        ),
        "edited_text": (
            "For a period of thirty-six (36) months following the Date of Termination, the Company shall, "
            "at its cost, provide Employee with health insurance benefits substantially similar to those "
            "in effect immediately prior to termination [E1]."
        ),
    },
    {
        "section_id": "seed_5",
        "section_title": "Misconduct",
        "original_text": (
            "If the employee does something seriously wrong, they can be fired for misconduct "
            "and won't get severance."
        ),
        "edited_text": (
            "In the event of termination for Misconduct, the Company's sole obligation shall be "
            "payment of any unpaid Base Compensation accrued through the Date of Termination [E5]. "
            "No severance or additional benefits shall be payable."
        ),
    },
]


# =============================================================================
# Bootstrap helpers
# =============================================================================

def _get_document_count(client: httpx.Client) -> tuple[int, str | None]:
    """Return (count, first_document_id_or_None)."""
    r = client.get("/documents")
    r.raise_for_status()
    data = r.json()
    docs = data.get("documents", [])
    first_id = docs[0]["document_id"] if docs else None
    return data["count"], first_id


def _ingest_pdf(client: httpx.Client, pdf_path: Path) -> tuple[str, str]:
    """Upload a PDF and return (job_id, document_id)."""
    with open(pdf_path, "rb") as fh:
        # httpx multipart: files={"file": (filename, file_object, content_type)}
        r = client.post(
            "/upload",
            files={"file": (pdf_path.name, fh, "application/pdf")},
        )
    r.raise_for_status()
    data = r.json()
    return data["job_id"], data["document_id"]


def _poll_job(client: httpx.Client, job_id: str, timeout: int = 300) -> dict:
    """
    Poll GET /job/{job_id} until status is 'completed' or 'failed'.

    The ingestion pipeline runs in the background (embed + Qdrant store can
    take 10–60 seconds depending on document length and CPU speed).
    We print dots so the user knows it's working.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/job/{job_id}")
        r.raise_for_status()
        job = r.json()
        status = job["status"]
        stage  = job.get("stage", "")
        progress = job.get("progress", 0)

        if status == "done":
            print(f"\r  Job {job_id[:8]}… completed ({stage})          ")
            return job
        if status == "failed":
            raise RuntimeError(f"Ingestion job failed: {job.get('error')}")

        print(f"\r  Ingesting… {stage} ({progress}%)   ", end="", flush=True)
        time.sleep(4)

    raise TimeoutError(f"Ingestion job {job_id} did not complete within {timeout}s")


def _generate_draft(client: httpx.Client, document_id: str) -> dict:
    """POST /draft and return the full response dict."""
    r = client.post(
        "/draft",
        json={
            "document_id": document_id,
            "query": "Summarize the compensation, termination, and severance terms",
            "draft_type": "case_fact_summary",
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def _save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved → {path.relative_to(path.parent.parent.parent)}")


def _wait_for_patterns(edit_ids: list[str], timeout: int = 90) -> None:
    """
    Poll SQLite until every edit's pattern_extraction_status is no longer 'pending'.

    Each edit requires two Groq calls (classification + rule extraction), which
    takes ~3–5 s each.  We give it up to `timeout` seconds total.
    """
    from python_service.db.session import engine
    from python_service.db.models import Edit

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        with Session(engine) as session:
            pending = [
                e
                for eid in edit_ids
                for e in [session.get(Edit, eid)]
                if e and e.pattern_extraction_status == "pending"
            ]
        if not pending:
            return
        print(f"  {len(pending)} edit(s) still extracting patterns…")

    print("  Warning: some edits may still be processing (timeout reached)")


def _get_most_recent_draft_id() -> str | None:
    """Query SQLite directly for the newest draft_id."""
    from python_service.db.session import engine
    from python_service.db.models import Draft

    with Session(engine) as session:
        drafts = session.exec(select(Draft).order_by(Draft.created_at.desc())).all()
    return drafts[0].draft_id if drafts else None


# =============================================================================
# Main entry point
# =============================================================================

def seed(base_url: str = BASE_URL) -> None:
    # Use a long timeout because ingestion + generation both hit local models
    client = httpx.Client(base_url=base_url, timeout=120)

    print("PSL Seed Script")
    print("=" * 60)

    # ── Step 1: Check whether any documents exist ─────────────────────────────
    # If the database is empty we auto-ingest the example contract PDF so the
    # user doesn't have to do any manual setup steps.
    doc_count, document_id = _get_document_count(client)

    if doc_count == 0:
        # ── Bootstrap: auto-ingest the example PDF ────────────────────────────
        if not SEED_PDF.exists():
            print(
                f"  ERROR: No documents found and example PDF is missing at {SEED_PDF}.\n"
                f"  Run `python -m scripts.generate_examples` first to create it."
            )
            return

        print(f"\n[Bootstrap] No documents found. Auto-ingesting {SEED_PDF.name}…")
        job_id, document_id = _ingest_pdf(client, SEED_PDF)
        print(f"  Job started: {job_id}  Document: {document_id}")

        # Poll until the pipeline finishes (embed → Qdrant store)
        _poll_job(client, job_id)
        print(f"  Ingestion complete. document_id = {document_id}")

    else:
        print(f"\nFound {doc_count} existing document(s). Using: {document_id}")

    # ── Step 2: Generate a BASELINE draft (no patterns in system yet) ─────────
    # We save this as the "before" benchmark for the improvement report.
    print("\n[Step 2] Generating baseline draft (no patterns applied)…")
    baseline = _generate_draft(client, document_id)

    if baseline.get("status") not in ("ok",):
        print(f"  WARNING: Draft returned status={baseline.get('status')!r} — {baseline}")
    else:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        _save_json(baseline, OUTPUTS / "draft_baseline.json")
        print(
            f"  Baseline: grounding={baseline['grounding_score']:.3f}  "
            f"overall_judge={baseline['judge_scores'].get('overall', '?')}  "
            f"patterns_applied={baseline['patterns_applied']}"
        )

    draft_id = baseline.get("draft_id") or _get_most_recent_draft_id()
    if not draft_id:
        print("  ERROR: Could not obtain a draft_id. Aborting.")
        return

    # ── Step 3: Submit the seed edits ─────────────────────────────────────────
    # Each edit is a before/after pair representing an operator's correction.
    # The background task extracts a generalised rule ("pattern") from each pair.
    print(f"\n[Step 3] Submitting {len(SEED_EDITS)} seed edits to draft {draft_id[:8]}…")
    r = client.post("/feedback", json={"draft_id": draft_id, "edits": SEED_EDITS})
    r.raise_for_status()
    result   = r.json()
    edit_ids = result.get("edit_ids", [])
    print(f"  Submitted. edit_ids: {[e[:8] for e in edit_ids]}")

    # ── Step 4: Wait for pattern extraction ───────────────────────────────────
    print("\n[Step 4] Waiting for pattern extraction (Groq Llama 3.3 70B)…")
    _wait_for_patterns(edit_ids)
    print("  All edits processed.")

    # ── Step 5: Generate an IMPROVED draft (patterns now in system) ───────────
    # This draft will have `patterns_applied > 0` and should score higher.
    print("\n[Step 5] Generating improved draft (with learned patterns)…")
    improved = _generate_draft(client, document_id)

    if improved.get("status") not in ("ok",):
        print(f"  WARNING: Draft returned status={improved.get('status')!r}")
    else:
        _save_json(improved, OUTPUTS / "draft_improved.json")
        print(
            f"  Improved: grounding={improved['grounding_score']:.3f}  "
            f"overall_judge={improved['judge_scores'].get('overall', '?')}  "
            f"patterns_applied={improved['patterns_applied']}"
        )

    # ── Step 6: Report patterns ───────────────────────────────────────────────
    from python_service.db.models import Pattern
    from python_service.db.session import engine

    with Session(engine) as session:
        patterns = session.exec(select(Pattern).where(Pattern.is_active == True)).all()

    print(f"\n[Done] Active patterns in system: {len(patterns)}")
    for p in patterns:
        print(f"  [{p.rule_type}] freq={p.frequency} conf={p.confidence:.2f}  {p.description[:70]}")

    print("\nNext steps:")
    print("  GET /patterns                     — view all patterns")
    print("  GET /evaluation/improvement-report — view before/after delta")


if __name__ == "__main__":
    seed()
