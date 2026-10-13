"""
Graph definition for the multi-agent orchestrator.

Builds and compiles the LangGraph StateGraph that wires together
the Supervisor, Research Agent, and Analysis Agent nodes.

Topology:
    START → Supervisor → (research | analyst | END)
    research → Supervisor
    analyst  → Supervisor
"""

from langgraph.graph import StateGraph, END, START

from state import OrchestratorState
from agents.research_agent import research_node
from agents.analyst_agent import analyst_node
from agents.supervisor import supervisor_node, route_supervisor


def build_graph(checkpointer=None):
    """
    Construct and compile the orchestrator graph.

    Args:
        checkpointer: An optional LangGraph checkpointer for state persistence
                      (e.g., MemorySaver for in-memory conversation history).

    Returns:
        A compiled LangGraph application ready to invoke or stream.
    """
    workflow = StateGraph(OrchestratorState)

    # --- Register nodes ---
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("analyst", analyst_node)

    # --- Entry point ---
    workflow.add_edge(START, "supervisor")

    # --- Supervisor conditional routing ---
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research": "research",
            "analyst": "analyst",
            END: END,
        },
    )

    # --- Agent feedback edges (both loop back to supervisor) ---
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("analyst", "supervisor")

    return workflow.compile(checkpointer=checkpointer)


# Module-level default app instance (no checkpointer)
app = build_graph()
