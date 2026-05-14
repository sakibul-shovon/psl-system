"""
Pattern decay — auto-archive stale or low-consensus patterns.

WHY do we need this?
Every time an operator makes an unusual edit, the system extracts a pattern.
Some of those are genuine recurring rules; others are one-off typo fixes or
context-specific rewrites that should never generalise. Without decay, these
accumulate and clutter the pattern store, adding noise to every future draft.

Two decay rules:
  1. FREQUENCY=1 + no reinforcement in 60 days → archive
     (Nobody else ever repeated this edit. It's probably a one-off.)
  2. operator_consensus < 0.3 + frequency >= 5 → flag for review
     (Multiple operators reinforced this, but most don't agree on it.
      Suspicious — may be contradictory or ambiguous.)

Safe to run on a schedule (e.g., daily cron). Idempotent — already-inactive
patterns are skipped.

Usage:
    python -m scripts.prune_patterns
    python -m scripts.prune_patterns --dry-run   (show what WOULD be pruned)
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add psl-system root to path so `python -m scripts.prune_patterns` works
# when run from inside the psl-system/ directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlmodel import Session, select

from python_service.db.models import Pattern
from python_service.db.session import create_db_and_tables, engine
from python_service.vector.qdrant_store import qdrant_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STALE_DAYS = 60          # patterns untouched for this long are candidates
STALE_MIN_FREQ = 1       # only auto-archive if frequency is still at 1
LOW_CONSENSUS_THRESHOLD = 0.30
LOW_CONSENSUS_MIN_FREQ  = 5   # only flag consensus issues when well-sampled


def prune(dry_run: bool = False) -> dict:
    """
    Run the decay rules and archive/flag patterns.

    Returns a summary dict: {archived, flagged, skipped}.
    """
    create_db_and_tables()

    cutoff = datetime.utcnow() - timedelta(days=STALE_DAYS)

    with Session(engine) as session:
        active_patterns = session.exec(
            select(Pattern).where(Pattern.is_active == True)
        ).all()

    archived = []
    flagged  = []
    skipped  = 0

    for p in active_patterns:
        # Rule 1: stale single-reinforcement pattern
        if p.frequency <= STALE_MIN_FREQ and p.last_reinforced_at < cutoff:
            age_days = (datetime.utcnow() - p.last_reinforced_at).days
            logger.info(
                "[STALE] pattern %s | freq=%d | age=%d days | %s",
                p.pattern_id[:8], p.frequency, age_days, p.description[:60],
            )
            archived.append(p.pattern_id)

            if not dry_run:
                with Session(engine) as session:
                    row = session.get(Pattern, p.pattern_id)
                    if row:
                        row.is_active = False
                        session.add(row)
                        session.commit()

                # Remove from Qdrant so it no longer pollutes retrieval
                if p.qdrant_point_id:
                    try:
                        qdrant_store.update_pattern_payload(
                            p.qdrant_point_id,
                            {"is_active": False},
                        )
                    except Exception as exc:
                        logger.warning(
                            "Qdrant payload update failed for %s: %s",
                            p.pattern_id[:8], exc,
                        )

        # Rule 2: low-consensus, high-frequency pattern → flag for human review
        elif p.frequency >= LOW_CONSENSUS_MIN_FREQ and p.operator_consensus < LOW_CONSENSUS_THRESHOLD:
            logger.warning(
                "[LOW CONSENSUS] pattern %s | freq=%d | consensus=%.2f | %s",
                p.pattern_id[:8], p.frequency, p.operator_consensus, p.description[:60],
            )
            flagged.append(p.pattern_id)

        else:
            skipped += 1

    summary = {
        "archived": len(archived),
        "flagged":  len(flagged),
        "skipped":  skipped,
        "dry_run":  dry_run,
        "archived_ids": archived,
        "flagged_ids":  flagged,
    }

    verb = "Would archive" if dry_run else "Archived"
    logger.info(
        "%s %d pattern(s), flagged %d for review, skipped %d healthy pattern(s).",
        verb, len(archived), len(flagged), skipped,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune stale/low-consensus patterns.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be pruned without actually archiving.",
    )
    args = parser.parse_args()

    result = prune(dry_run=args.dry_run)

    if args.dry_run:
        print("\nDRY RUN — no changes made.")
    print(f"\nSummary: {result}")


if __name__ == "__main__":
    main()
