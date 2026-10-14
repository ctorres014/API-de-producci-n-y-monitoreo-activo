"""
Observability setup for the multi-agent API.

Supports two backends:
  - Arize Phoenix (default): set PHOENIX_ENABLED=true
  - LangSmith (alternative): set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY

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
    """
    Enable Arize Phoenix tracing when PHOENIX_ENABLED=true.

    If PHOENIX_COLLECTOR_ENDPOINT is set, traces are sent to an already running
    Phoenix server (`phoenix serve`). Otherwise an embedded server is launched
    in-process. Any failure is logged and the API keeps running without tracing.
    """
    if os.getenv("PHOENIX_ENABLED", "false").lower() != "true":
        return False

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()

    try:
        import phoenix as px  # type: ignore
        from openinference.instrumentation.langchain import LangChainInstrumentor  # type: ignore
        from phoenix.otel import register  # type: ignore

        if not endpoint:
            px.launch_app()  # start embedded Phoenix server on port 6006
        tracer_provider = register()  # reads PHOENIX_COLLECTOR_ENDPOINT
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info(
            "Arize Phoenix tracing enabled → dashboard at %s",
            endpoint or "http://localhost:6006",
        )
        return True

    except ImportError as exc:
        logger.warning(
            "PHOENIX_ENABLED=true but packages are missing: %s. "
            "Install: pip install arize-phoenix openinference-instrumentation-langchain",
            exc,
        )
        return False

    except Exception as exc:
        logger.warning(
            "Arize Phoenix could not be started (%s) — continuing without tracing. "
            "Run `phoenix serve` in another terminal and set "
            "PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006",
            exc,
        )
        return False
