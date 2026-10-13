"""
Observability setup for the multi-agent API.

Supports two backends:
  - LangSmith (default): set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY
  - Arize Phoenix (optional): set PHOENIX_ENABLED=true

Both are configured via environment variables so no code changes are needed
to switch between them.
"""

import logging
import os

logger = logging.getLogger(__name__)


def init_observability() -> str:
    """
    Initialize the active observability backend.

    Returns the name of the backend that was configured, or "none" if
    tracing is disabled.
    """
    if _try_phoenix():
        return "phoenix"
    if _try_langsmith():
        return "langsmith"
    logger.warning(
        "No observability backend configured. "
        "Set LANGCHAIN_TRACING_V2=true or PHOENIX_ENABLED=true."
    )
    return "none"


def _try_langsmith() -> bool:
    """Activate LangSmith tracing when the required env vars are present."""
    api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()

    if not api_key or tracing != "true":
        return False

    project = os.getenv("LANGCHAIN_PROJECT", "entrega7-multi-agent")
    os.environ["LANGCHAIN_PROJECT"] = project
    logger.info("LangSmith tracing enabled → project: %s", project)
    return True


def _try_phoenix() -> bool:
    """Launch an in-process Arize Phoenix collector when PHOENIX_ENABLED=true."""
    if os.getenv("PHOENIX_ENABLED", "false").lower() != "true":
        return False

    try:
        import phoenix as px  # type: ignore
        from openinference.instrumentation.langchain import LangChainInstrumentor  # type: ignore
        from opentelemetry import trace as otel_trace  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore
        from phoenix.otel import register  # type: ignore

        tracer_provider = register()
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Arize Phoenix tracing enabled → dashboard at http://localhost:6006")
        return True

    except ImportError as exc:
        logger.warning(
            "PHOENIX_ENABLED=true but packages are missing: %s. "
            "Install: pip install arize-phoenix openinference-instrumentation-langchain",
            exc,
        )
        return False
