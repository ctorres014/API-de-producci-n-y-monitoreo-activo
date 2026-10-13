"""
Background worker that executes the LangGraph orchestrator for a given job.

Each job transitions through these Redis-persisted states:

    PENDING → RUNNING → DONE
                     ↘ AWAITING_APPROVAL → RUNNING → DONE
                     ↘ FAILED

The worker is called as a FastAPI BackgroundTask (runs in a thread pool
managed by asyncio's run_in_executor), so it never blocks the event loop.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import redis as redis_lib
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Allow root-level imports (state.py, config.py, agents/, etc.)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# -----------------------------------------------------------------
# Job status constants
# -----------------------------------------------------------------
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_AWAITING = "awaiting_approval"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_REJECTED = "rejected"

_JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "86400"))  # 24 h


# -----------------------------------------------------------------
# Redis helpers
# -----------------------------------------------------------------

def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def create_job(redis_client: redis_lib.Redis, query: str) -> str:
    """Create a new job record in Redis and return the job_id."""
    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "status": STATUS_PENDING,
        "query": query,
        "result": None,
        "error": None,
        "interrupt_value": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    redis_client.setex(_job_key(job_id), _JOB_TTL_SECONDS, json.dumps(payload))
    return job_id


def get_job(redis_client: redis_lib.Redis, job_id: str) -> Optional[dict]:
    """Return the job dict or None if not found."""
    raw = redis_client.get(_job_key(job_id))
    if raw is None:
        return None
    return json.loads(raw)


def update_job(redis_client: redis_lib.Redis, job_id: str, **fields) -> None:
    """Merge `fields` into the existing job record."""
    job = get_job(redis_client, job_id)
    if job is None:
        logger.error("update_job: job %s not found in Redis", job_id)
        return
    job.update(fields)
    job["updated_at"] = _now()
    redis_client.setex(_job_key(job_id), _JOB_TTL_SECONDS, json.dumps(job))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------
# Initial state builder
# -----------------------------------------------------------------

def _build_initial_state(query: str) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "next_agent": "research",
        "research_data": None,
        "analysis_result": None,
        "task_completed": False,
        "iteration_count": 0,
        "agent_contributions": {},
        "validation_notes": None,
        "is_critical": False,
        "is_approved": None,
    }


def _serialize_result(state: dict) -> dict:
    """Extract a serializable summary from the final graph state."""
    return {
        "research_data": state.get("research_data"),
        "analysis_result": state.get("analysis_result"),
        "agent_contributions": state.get("agent_contributions", {}),
        "validation_notes": state.get("validation_notes"),
        "iteration_count": state.get("iteration_count", 0),
        "task_completed": state.get("task_completed", False),
        "is_critical": state.get("is_critical", False),
        "is_approved": state.get("is_approved"),
    }


# -----------------------------------------------------------------
# Core worker functions (called from BackgroundTasks)
# -----------------------------------------------------------------

def run_graph(
    redis_client: redis_lib.Redis,
    graph: Any,
    job_id: str,
    query: str,
) -> None:
    """
    Execute the orchestrator graph for `job_id` and persist the outcome.

    This function is synchronous — it is called inside asyncio's default
    thread pool via `loop.run_in_executor` so the FastAPI event loop is
    never blocked.
    """
    config = {"configurable": {"thread_id": job_id}}

    update_job(redis_client, job_id, status=STATUS_RUNNING)
    logger.info("Job %s started", job_id)

    try:
        initial_state = _build_initial_state(query)
        final_state = graph.invoke(initial_state, config=config)

        result = _serialize_result(final_state)
        update_job(redis_client, job_id, status=STATUS_DONE, result=result)
        logger.info("Job %s completed", job_id)

    except Exception as exc:
        # Check for LangGraph interrupt (HITL pause)
        exc_name = type(exc).__name__
        if "GraphInterrupt" in exc_name or "Interrupt" in exc_name:
            _handle_interrupt(redis_client, graph, job_id, exc)
        else:
            logger.exception("Job %s failed: %s", job_id, exc)
            update_job(redis_client, job_id, status=STATUS_FAILED, error=str(exc))


def _handle_interrupt(
    redis_client: redis_lib.Redis,
    graph: Any,
    job_id: str,
    exc: Exception,
) -> None:
    """
    Handle a GraphInterrupt exception by parking the job as AWAITING_APPROVAL.

    Extracts the interrupt value (the dict passed to interrupt()) and stores
    it in Redis so the GET /tasks/{id} response can surface it to the client.
    """
    # The interrupt value is attached to the exception as .args[0] or .value
    interrupt_value: Any = None
    if hasattr(exc, "args") and exc.args:
        raw = exc.args[0]
        # LangGraph wraps the value in a list of Interrupt objects
        if isinstance(raw, (list, tuple)) and raw:
            item = raw[0]
            interrupt_value = getattr(item, "value", str(item))
        else:
            interrupt_value = raw

    # Attempt to make it JSON-serialisable
    try:
        json.dumps(interrupt_value)
    except (TypeError, ValueError):
        interrupt_value = str(interrupt_value)

    update_job(
        redis_client,
        job_id,
        status=STATUS_AWAITING,
        interrupt_value=interrupt_value,
    )
    logger.info("Job %s awaiting human approval", job_id)


def resume_graph(
    redis_client: redis_lib.Redis,
    graph: Any,
    job_id: str,
    approved: bool,
) -> None:
    """
    Resume a paused job after human approval/rejection.

    Called as a BackgroundTask from POST /tasks/{id}/approve.
    """
    from langgraph.types import Command  # type: ignore

    config = {"configurable": {"thread_id": job_id}}
    update_job(redis_client, job_id, status=STATUS_RUNNING)
    logger.info("Job %s resuming with approved=%s", job_id, approved)

    try:
        final_state = graph.invoke(
            Command(resume={"approved": approved}),
            config=config,
        )
        result = _serialize_result(final_state)
        status = STATUS_DONE if approved else STATUS_REJECTED
        update_job(redis_client, job_id, status=status, result=result)
        logger.info("Job %s finished after HITL (approved=%s)", job_id, approved)

    except Exception as exc:
        logger.exception("Job %s failed during resume: %s", job_id, exc)
        update_job(redis_client, job_id, status=STATUS_FAILED, error=str(exc))
