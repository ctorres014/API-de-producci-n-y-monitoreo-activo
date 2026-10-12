"""
API-facing graph builder.

Extends the base multi-agent graph (M6) with:
  - RedisSaver checkpointer for durable state across HTTP requests
  - Human-in-the-Loop (HITL) node inserted before the analyst for critical tasks
  - Criticality detection that overrides the supervisor's routing decision

Graph topology:

    START
      │
      ▼
    supervisor ──► research ──► supervisor
      │
      ├──► human_approval ──(approved)──► analyst ──► supervisor
      │           │
      │        (rejected)
      │           │
      └──► analyst ──► supervisor
      │
      └──► END

The supervisor decides between research / analyst / FINISH as before.
After research completes and the supervisor next routes to analyst,
`route_supervisor_api` intercepts: if the query is critical and not yet
approved, it redirects to `human_approval` instead.
"""

import logging
import os
import sys

# Allow imports from the repo root (agents/, tools/, state.py, config.py)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from langgraph.graph import StateGraph, END, START  # noqa: E402

from state import OrchestratorState  # noqa: E402
from agents.research_agent import research_node  # noqa: E402
from agents.analyst_agent import analyst_node  # noqa: E402
from agents.supervisor import supervisor_node  # noqa: E402
from app.hitl import human_approval_node, route_after_approval, is_critical_query  # noqa: E402

logger = logging.getLogger(__name__)


def route_supervisor_api(state: OrchestratorState) -> str:
    """
    Extended routing function for the API graph.

    Behaves like the base route_supervisor but adds HITL interception:
    when the supervisor wants to run the analyst and the query is critical
    (and not yet approved), redirect to human_approval instead.
    """
    next_agent = state.get("next_agent", "FINISH")

    if next_agent == "research":
        return "research"

    if next_agent == "analyst":
        # Intercept: send critical, unapproved tasks to HITL
        if is_critical_query(state.get("user_query", "")) and not state.get("is_approved"):
            logger.info("Critical query detected — redirecting to human_approval node")
            return "human_approval"
        return "analyst"

    return END


def build_api_graph(checkpointer=None):
    """
    Build and compile the orchestrator graph with HITL support.

    Args:
        checkpointer: A LangGraph checkpointer (e.g. RedisSaver).
                      When None, state is not persisted between calls.

    Returns:
        A compiled LangGraph application.
    """
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("human_approval", human_approval_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor_api,
        {
            "research": "research",
            "analyst": "analyst",
            "human_approval": "human_approval",
            END: END,
        },
    )

    # After HITL node, route to analyst (approved) or END (rejected)
    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "analyst": "analyst",
            END: END,
        },
    )

    workflow.add_edge("research", "supervisor")
    workflow.add_edge("analyst", "supervisor")

    return workflow.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])


def get_redis_checkpointer():
    """
    Create a RedisSaver instance from REDIS_URL.

    Returns None (falling back to in-memory) if the package is not installed
    or Redis is unreachable — avoids crashing the entire API on startup when
    Redis is temporarily down.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        from langgraph.checkpoint.redis import RedisSaver  # type: ignore
        import redis as redis_lib  # type: ignore

        client = redis_lib.from_url(redis_url, decode_responses=False)
        client.ping()  # fast connectivity check
        saver = RedisSaver(client)
        saver.setup()
        logger.info("RedisSaver connected → %s", redis_url)
        return saver

    except ImportError:
        logger.warning(
            "langgraph-checkpoint-redis not installed — falling back to MemorySaver. "
            "Install with: pip install langgraph-checkpoint-redis"
        )
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — falling back to MemorySaver.", exc)

    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
    return MemorySaver()
