"""
In-memory job tracker for background pipeline runs.

When a document is uploaded, FastAPI's BackgroundTasks runs the ingestion
pipeline asynchronously. The route returns a jobId immediately; the UI
polls GET /job/{id} to show live progress.

This is a simple dict — no Redis, no queue. For a single-user demo on one
machine, this is sufficient. On restart all job state is lost (fine for demo).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Valid job statuses
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class Job:
    job_id: str
    document_id: Optional[str] = None
    status: str = STATUS_PENDING
    stage: str = ""            # human-readable current step, e.g. "chunking"
    progress: int = 0          # 0–100 percentage estimate
    result: Optional[Any] = None   # populated on success (e.g. {draftId, chunks})
    error: Optional[str] = None    # populated on failure
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# The global job registry. All reads and writes happen in the same process,
# so there's no concurrency issue with a plain dict.
_jobs: dict[str, Job] = {}


def create_job(job_id: str, document_id: Optional[str] = None) -> Job:
    job = Job(job_id=job_id, document_id=document_id)
    _jobs[job_id] = job
    logger.info("Job created: %s", job_id)
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[int] = None,
    result: Optional[Any] = None,
    error: Optional[str] = None,
    document_id: Optional[str] = None,
) -> None:
    job = _jobs.get(job_id)
    if job is None:
        logger.warning("update_job called for unknown job_id: %s", job_id)
        return
    if status is not None:
        job.status = status
    if stage is not None:
        job.stage = stage
    if progress is not None:
        job.progress = progress
    if result is not None:
        job.result = result
    if error is not None:
        job.error = error
    if document_id is not None:
        job.document_id = document_id
    job.updated_at = datetime.utcnow()


def fail_job(job_id: str, error: str) -> None:
    update_job(job_id, status=STATUS_FAILED, error=error, progress=0)
    logger.error("Job %s failed: %s", job_id, error)
