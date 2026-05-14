"""
Central configuration loader.

Reads environment variables from .env (via pydantic-settings) and exposes them
as a typed `settings` singleton. Every other module imports `settings` from here
instead of calling os.getenv directly — that way the entire app has one place
where configuration is defined, validated, and documented.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Values come from .env, falling back to defaults below."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,    # GEMINI_API_KEY in .env maps to gemini_api_key here
        extra="ignore",          # ignore unknown keys in .env instead of erroring
    )

    # ─── LLM API keys ────────────────────────────────────────────────────
    # Default to empty strings so the app can boot even before keys are set
    # (the /health endpoint still works). Endpoints that actually call LLMs
    # will raise a clear error if a key is missing.
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # ─── Service URLs ────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""   # required for Qdrant Cloud; empty = no auth (local)
    database_url: str = "sqlite:///./data/psl.db"

    # ─── OCR ─────────────────────────────────────────────────────────────
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # ─── File locations (relative to project root) ───────────────────────
    bm25_dir: Path = Path("./data/bm25")
    uploads_dir: Path = Path("./data/uploads")

    # ─── Demo defaults ───────────────────────────────────────────────────
    default_operator_id: str = "op_harvey"

    # ─── Optional: Langfuse (Friday stretch goal) ────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


# Module-level singleton. Import this from anywhere:
#   from python_service.config import settings
#   client = SomeClient(api_key=settings.gemini_api_key)
settings = Settings()
