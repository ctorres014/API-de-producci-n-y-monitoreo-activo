"""
State definition for the multi-agent orchestrator.

OrchestratorState is the shared state passed between all nodes in the graph.
It tracks messages, agent outputs, routing decisions, and metadata.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict):
    """
    Shared state for the multi-agent orchestrator graph.

    Fields:
        messages: Conversation history managed by LangGraph's add_messages reducer.
        user_query: The original user query string for easy access by all agents.
        next_agent: The name of the next agent to route to ("research", "analyst", "FINISH").
        research_data: Raw research findings produced by the Research Agent.
        analysis_result: Analytical insights produced by the Analysis Agent.
        task_completed: Flag indicating the full pipeline has finished.
        iteration_count: Number of supervisor decisions made (guards against infinite loops).
        agent_contributions: Dict mapping agent names to their output summaries.
        validation_notes: Optional notes from the supervisor about result quality.
        is_critical: Whether the task requires human approval before analysis.
        is_approved: Whether a human has approved the task (None = not yet decided).
    """

    messages: Annotated[list, add_messages]
    user_query: str
    next_agent: str
    research_data: Optional[str]
    analysis_result: Optional[str]
    task_completed: bool
    iteration_count: int
    agent_contributions: dict  # e.g. {"research": "...", "analyst": "..."}
    validation_notes: Optional[str]
    is_critical: bool
    is_approved: Optional[bool]
