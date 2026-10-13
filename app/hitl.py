"""
Human-in-the-Loop (HITL) support.

Provides:
  - is_critical_query(): detects whether a query triggers approval flow
  - human_approval_node(): LangGraph node that calls interrupt() to pause the graph
  - route_after_approval(): edge function that routes post-approval to analyst or END
"""

import logging
import re

from langgraph.graph import END
from langgraph.types import interrupt

logger = logging.getLogger(__name__)

# Keywords that flag a query as requiring human approval before the analyst runs.
# These represent operations with potential side-effects or high cost.
_CRITICAL_PATTERNS = re.compile(
    r"\b(deploy|delete|remove|drop|execute|run script|publish|send email|"
    r"purchase|buy|payment|charge|transfer|update (database|db|prod)|"
    r"overwrite|truncate|wipe|format|reset|terminate|kill process)\b",
    re.IGNORECASE,
)


def is_critical_query(user_query: str) -> bool:
    """Return True if the query matches patterns that require human approval."""
    return bool(_CRITICAL_PATTERNS.search(user_query or ""))


def human_approval_node(state: dict) -> dict:
    """
    LangGraph node that pauses execution and waits for human approval.

    When the graph reaches this node:
      1. LangGraph saves the full checkpoint to Redis.
      2. `interrupt()` raises GraphInterrupt, surfacing the approval request
         to the caller (worker.py catches it and sets job status to
         AWAITING_APPROVAL in Redis).
      3. When the operator POSTs to /tasks/{id}/approve, the worker resumes
         the graph via Command(resume={...}), which returns the human's
         decision here as `human_response`.
      4. The node writes `is_approved` back into state so the routing edge
         can decide whether to continue to the analyst or terminate.
    """
    user_query = state.get("user_query", "")
    research_data = state.get("research_data", "")

    approval_request = {
        "message": (
            "This task has been flagged as critical and requires human approval "
            "before the Analysis Agent runs."
        ),
        "query": user_query,
        "research_summary": (research_data[:600] + "…") if research_data and len(research_data) > 600 else (research_data or "No research data yet"),
        "instructions": (
            "POST /tasks/{job_id}/approve with body "
            '{"approved": true} to continue or {"approved": false} to abort.'
        ),
    }

    logger.info("HITL: interrupting for human approval on query=%r", user_query[:80])

    # Pauses the graph; returns the value passed to Command(resume=...)
    human_response = interrupt(approval_request)

    approved = (
        human_response.get("approved", False)
        if isinstance(human_response, dict)
        else bool(human_response)
    )

    logger.info("HITL: resumed with approved=%s", approved)

    return {
        "is_approved": approved,
        "validation_notes": (
            "Human approved — proceeding to analysis."
            if approved
            else "Human rejected — aborting analysis."
        ),
    }


def route_after_approval(state: dict) -> str:
    """
    Edge function after human_approval_node.

    Routes to 'analyst' when approved, otherwise ends the graph.
    """
    if state.get("is_approved"):
        return "analyst"
    return END
