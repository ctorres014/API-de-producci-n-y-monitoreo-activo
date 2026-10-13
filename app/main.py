"""
FastAPI entry point for the multi-agent orchestrator API.

Endpoints:
    POST  /tasks                  Submit a new task (async, returns job_id immediately)
    GET   /tasks/{job_id}         Poll the status of a running or completed job
    POST  /tasks/{job_id}/approve Resume a HITL-paused job (approve or reject)
    GET   /health                 Liveness check
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import redis as redis_lib
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# Allow root-level imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"))

from app.observability import init_observability  # noqa: E402
from app.graph import build_api_graph, get_redis_checkpointer  # noqa: E402
from app.worker import (  # noqa: E402
    create_job,
    get_job,
    run_graph,
    resume_graph,
    STATUS_AWAITING,
    STATUS_PENDING,
    STATUS_FAILED,
    STATUS_DONE,
    STATUS_REJECTED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Application state (initialised in lifespan)
# -----------------------------------------------------------------
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup; close them on shutdown."""
    obs_backend = init_observability()
    logger.info("Observability backend: %s", obs_backend)

    checkpointer = get_redis_checkpointer()
    graph = build_api_graph(checkpointer=checkpointer)

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis_lib.from_url(redis_url, decode_responses=True)

    _state["graph"] = graph
    _state["redis"] = redis_client
    logger.info("API ready — graph compiled, Redis connected")

    yield  # ← application serves requests here

    redis_client.close()
    logger.info("API shutdown complete")


app = FastAPI(
    title="Multi-Agent Orchestrator API",
    version="2.0.0",
    description=(
        "Async REST API for the LangGraph multi-agent orchestrator "
        "(Entrega 7 — CoderHouse IA Engineering)."
    ),
    lifespan=lifespan,
)


# -----------------------------------------------------------------
# Request / Response models
# -----------------------------------------------------------------

class TaskRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The question or task to process")


class TaskResponse(BaseModel):
    job_id: str
    status: str
    message: str = ""


class ApprovalRequest(BaseModel):
    approved: bool = Field(..., description="True to approve, False to reject")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    query: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    interrupt_value: Optional[object] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# -----------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------

@app.post("/tasks", response_model=TaskResponse, status_code=202)
async def submit_task(body: TaskRequest, background_tasks: BackgroundTasks):
    """
    Submit a new task to the multi-agent pipeline.

    Returns immediately with a job_id. The graph runs in the background.
    Poll GET /tasks/{job_id} to track progress.
    """
    redis_client = _state["redis"]
    graph = _state["graph"]

    job_id = create_job(redis_client, body.query)
    logger.info("Enqueued job %s for query=%r", job_id, body.query[:80])

    # Run graph in threadpool so the event loop is never blocked
    background_tasks.add_task(
        run_in_threadpool, run_graph, redis_client, graph, job_id, body.query
    )

    return TaskResponse(
        job_id=job_id,
        status=STATUS_PENDING,
        message="Task enqueued. Poll GET /tasks/{job_id} for status.",
    )


@app.get("/tasks/{job_id}", response_model=JobStatusResponse)
async def get_task_status(job_id: str):
    """
    Return the current status and result of a job.

    Possible statuses:
      pending            — queued, not started yet
      running            — graph is executing
      awaiting_approval  — paused at HITL node, needs POST /tasks/{id}/approve
      done               — completed successfully
      rejected           — completed; human rejected at HITL
      failed             — graph raised an unhandled exception
    """
    redis_client = _state["redis"]
    job = get_job(redis_client, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatusResponse(**job)


@app.post("/tasks/{job_id}/approve", response_model=TaskResponse)
async def approve_task(job_id: str, body: ApprovalRequest, background_tasks: BackgroundTasks):
    """
    Resume a HITL-paused job with a human decision.

    Requires the job to be in `awaiting_approval` status.
    """
    redis_client = _state["redis"]
    graph = _state["graph"]

    job = get_job(redis_client, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] != STATUS_AWAITING:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job is in status '{job['status']}', "
                f"expected '{STATUS_AWAITING}'. "
                "Only paused jobs can be approved."
            ),
        )

    logger.info("Human decision for job %s: approved=%s", job_id, body.approved)

    background_tasks.add_task(
        run_in_threadpool, resume_graph, redis_client, graph, job_id, body.approved
    )

    action = "approved" if body.approved else "rejected"
    return TaskResponse(
        job_id=job_id,
        status="running",
        message=f"Job {action}. Execution resuming in background.",
    )


@app.get("/health")
async def health():
    """Liveness probe."""
    try:
        _state["redis"].ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "ok",
        "redis": "connected" if redis_ok else "unavailable",
        "graph": "compiled" if _state.get("graph") else "not ready",
    }
