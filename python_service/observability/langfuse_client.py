"""
Langfuse observability wrapper.

Langfuse is a free, open-source LLM tracing dashboard. When you set
LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env, every LLM call
in this codebase is automatically logged to https://cloud.langfuse.com
with: prompt text, response, token counts, latency, model name.

WHY is this useful?
  Without tracing, debugging a bad draft means guessing which LLM call went
  wrong. With Langfuse, you can open the dashboard, click on any draft, and
  see the exact prompt sent to Gemini and the exact response received.
  This is standard practice at AI companies — it's what "production-grade"
  looks like.

DESIGN: graceful no-op
  If langfuse is not installed, or if the keys are empty, this module
  provides a dummy `observe` decorator that does nothing. The rest of the
  codebase just uses `from python_service.observability.langfuse_client import observe`
  and never needs to check whether Langfuse is available.

SETUP (one-time):
  1. Go to https://cloud.langfuse.com → sign up free
  2. Create a project → copy the Public Key and Secret Key
  3. Add to your .env:
       LANGFUSE_PUBLIC_KEY=pk-lf-...
       LANGFUSE_SECRET_KEY=sk-lf-...
  4. pip install langfuse   (or: pip install -r requirements.txt)
  5. Restart the server — traces appear in the dashboard automatically
"""

import logging
import os

logger = logging.getLogger(__name__)

# ── Try to load Langfuse ───────────────────────────────────────────────────────
# We lazy-import to avoid a hard dependency. If the package isn't installed
# OR the API keys are not set, we fall back to a no-op decorator.

def _make_noop_observe():
    """Return a no-op @observe() decorator that passes functions through unchanged."""
    def observe(func=None, *, name=None, **kwargs):
        if func is not None:
            return func          # used as @observe (no parens)
        return lambda f: f       # used as @observe(...) (with parens)
    return observe


try:
    from langfuse.decorators import observe as _langfuse_observe   # type: ignore

    # Only activate if both keys are set — otherwise Langfuse would silently
    # fail on every call, adding latency for zero benefit.
    _pub  = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    _sec  = os.getenv("LANGFUSE_SECRET_KEY", "")
    _host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if _pub and _sec:
        observe = _langfuse_observe
        logger.info(
            "Langfuse tracing ENABLED → %s (project key: %s...)",
            _host, _pub[:8],
        )
    else:
        observe = _make_noop_observe()
        logger.info(
            "Langfuse keys not set — tracing DISABLED. "
            "Set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY in .env to enable."
        )

except ImportError:
    observe = _make_noop_observe()
    logger.info(
        "langfuse package not installed — tracing DISABLED. "
        "Run: pip install langfuse"
    )
