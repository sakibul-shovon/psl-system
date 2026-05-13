"""
Database engine and session management.

SQLite is file-based — the engine is a connection to data/psl.db.
`create_db_and_tables()` is called once at app startup to create any
tables that don't exist yet. It's safe to call repeatedly (idempotent).

FastAPI dependency `get_session` gives each request a fresh session
that auto-commits and auto-closes.
"""

import logging
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

from python_service.config import settings

logger = logging.getLogger(__name__)

# `check_same_thread=False` is required for SQLite when FastAPI's async
# workers access the same connection from different threads. SQLModel
# handles the per-request session lifecycle to keep this safe.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,   # set True to log every SQL statement (useful for debugging)
)


def create_db_and_tables() -> None:
    """
    Create all tables defined by SQLModel classes with `table=True`.
    Must be called before the first DB query — we call it in main.py's lifespan.
    """
    # This import causes SQLModel to register all table definitions
    import python_service.db.models  # noqa: F401 — side-effect import

    SQLModel.metadata.create_all(engine)
    logger.info("SQLite tables ready at %s", settings.database_url)


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a SQLModel session.

    Usage in a route:
        @app.post("/something")
        def handler(session: Session = Depends(get_session)):
            session.add(...)
            session.commit()
    """
    with Session(engine) as session:
        yield session
