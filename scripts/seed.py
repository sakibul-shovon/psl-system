"""
Seed script — pre-loads realistic operator edits so the system has learned
patterns from day one of a demo.

Run once after initial setup:
    python -m scripts.seed

What it does:
  1. Finds the most recent document in SQLite
  2. Generates a draft for it
  3. POSTs several realistic lawyer edits via /feedback
  4. Waits for background pattern extraction to finish
  5. Prints a summary of extracted patterns

Why seed instead of using real edits?
  The improvement-report endpoint needs drafts both before AND after patterns
  are applied to compute a delta.  Without seeding, a fresh install shows
  "not enough data."  Seeding gives the demo a meaningful baseline.
"""

import json
import time

import httpx
from sqlmodel import Session, select

BASE_URL = "http://localhost:8000"

# Realistic before/after edit pairs that cover different pattern types
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


def get_most_recent_draft_id(base_url: str) -> str | None:
    """Find the most recent draft_id by querying the database directly."""
    from python_service.db.session import engine
    from python_service.db.models import Draft

    with Session(engine) as session:
        drafts = session.exec(select(Draft).order_by(Draft.created_at.desc())).all()
    if not drafts:
        return None
    return drafts[0].draft_id


def seed(base_url: str = BASE_URL) -> None:
    client = httpx.Client(base_url=base_url, timeout=120)

    print("PSL Seed Script")
    print("=" * 50)

    # ── Step 1: Find a draft to attach edits to ───────────────────────────────
    draft_id = get_most_recent_draft_id(base_url)
    if not draft_id:
        print("No drafts found. Generate a draft first via POST /draft, then re-run seed.py")
        return

    print(f"Attaching seed edits to draft: {draft_id}")

    # ── Step 2: Submit the seed edits ─────────────────────────────────────────
    payload = {"draft_id": draft_id, "edits": SEED_EDITS}
    response = client.post("/feedback", json=payload)
    response.raise_for_status()
    result = response.json()
    edit_ids = result.get("edit_ids", [])
    print(f"Submitted {len(edit_ids)} seed edit(s). Waiting for pattern extraction...")

    # ── Step 3: Wait for background tasks to finish ───────────────────────────
    # Each edit takes ~3–5 seconds (two Groq calls). Poll until all are done.
    from python_service.db.session import engine
    from python_service.db.models import Edit

    max_wait = 60
    start = time.time()
    while time.time() - start < max_wait:
        time.sleep(3)
        with Session(engine) as session:
            pending = [
                e for eid in edit_ids
                for e in [session.get(Edit, eid)]
                if e and e.pattern_extraction_status == "pending"
            ]
        if not pending:
            break
        print(f"  {len(pending)} edit(s) still processing...")

    # ── Step 4: Report results ────────────────────────────────────────────────
    from python_service.db.models import Pattern

    with Session(engine) as session:
        patterns = session.exec(select(Pattern).where(Pattern.is_active == True)).all()

    print(f"\nDone. Active patterns in system: {len(patterns)}")
    for p in patterns:
        print(f"  [{p.rule_type}] {p.description[:70]}")

    print("\nRun GET /patterns to see all patterns.")
    print("Run GET /evaluation/improvement-report to see the improvement delta.")


if __name__ == "__main__":
    seed()
