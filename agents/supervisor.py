"""
Supervisor Agent for the multi-agent orchestrator.

The Supervisor is the central coordinator. It reads the current state,
decides whether to dispatch to the Research Agent, the Analysis Agent,
or to finish, and validates results before completing the pipeline.
"""

from typing import Optional, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from pydantic import BaseModel

from state import OrchestratorState
from config import get_llm, MAX_ITERATIONS


# ---------------------------------------------------------------------------
# Pydantic model for structured supervisor output
# ---------------------------------------------------------------------------

class SupervisorDecision(BaseModel):
    """Structured decision output from the Supervisor LLM."""

    next: Literal["research", "analyst", "FINISH"]
    reasoning: str
    validation_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Supervisor system prompt
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of a multi-agent pipeline. Your role is to orchestrate
a Research Agent and an Analysis Agent to answer the user's query completely and with high quality.

Decision rules (follow strictly):
1. If research_data is empty or None → dispatch to "research" first.
2. If research_data exists but analysis_result is empty or None → dispatch to "analyst".
3. If both research_data AND analysis_result are non-empty → apply the Validation Rubric below.
4. If neither agent has produced useful output after multiple iterations → return "FINISH" to avoid looping.

Validation Rubric (apply before returning FINISH):
- RELEVANCE: Does research_data directly address the user's query? If not → re-dispatch to "research".
- SOURCING: Does research_data cite sources, facts, or reference specific information? If completely generic → re-dispatch to "research".
- DEPTH: Does analysis_result produce concrete metrics, numbers, or specific conclusions? If it only paraphrases research without real analysis → re-dispatch to "analyst".
- COMPLETENESS: Does the combined output fully answer the user's question? If a critical aspect is missing → re-dispatch to the appropriate agent.
- If ALL rubric criteria pass → return "FINISH".

Always provide clear reasoning for your decision, citing which rubric criteria passed or failed.
Use validation_notes to communicate quality concerns to the next agent.

Respond ONLY with a JSON object matching this schema:
{
  "next": "research" | "analyst" | "FINISH",
  "reasoning": "<why you made this choice, citing rubric criteria>",
  "validation_notes": "<quality feedback for the next agent, or null>"
}"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def supervisor_node(state: OrchestratorState) -> dict:
    """
    LangGraph node for the Supervisor.

    Evaluates the current state and decides the next action.
    Forces FINISH when iteration_count >= MAX_ITERATIONS to prevent infinite loops.

    Args:
        state: The current OrchestratorState.

    Returns:
        A partial state dict with next_agent, validation_notes, and task_completed.
    """
    iteration_count = state.get("iteration_count", 0)
    user_query = state.get("user_query", "")
    research_data = state.get("research_data")
    analysis_result = state.get("analysis_result")

    # Hard stop: force FINISH to prevent infinite loops
    if iteration_count >= MAX_ITERATIONS:
        return {
            "next_agent": "FINISH",
            "task_completed": True,
            "validation_notes": (
                f"Maximum iterations ({MAX_ITERATIONS}) reached. "
                "Forcing task completion to prevent infinite loop."
            ),
        }

    # Build a context summary for the LLM
    context_parts = [
        f"User query: {user_query}",
        f"Current iteration: {iteration_count} / {MAX_ITERATIONS}",
        f"Research data available: {'YES' if research_data else 'NO'}",
        f"Analysis result available: {'YES' if analysis_result else 'NO'}",
    ]

    if research_data:
        preview = research_data[:500] + "..." if len(research_data) > 500 else research_data
        context_parts.append(f"\nResearch data preview:\n{preview}")

    if analysis_result:
        preview = analysis_result[:500] + "..." if len(analysis_result) > 500 else analysis_result
        context_parts.append(f"\nAnalysis result preview:\n{preview}")

    context_summary = "\n".join(context_parts)

    # Try to get a structured decision from the LLM
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(SupervisorDecision)

        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=context_summary),
        ]

        decision: SupervisorDecision = structured_llm.invoke(messages)
        next_agent = decision.next
        reasoning = decision.reasoning
        validation_notes = decision.validation_notes

    except Exception as exc:
        # Fallback: use rule-based logic if LLM call fails
        next_agent, reasoning, validation_notes = _rule_based_decision(
            research_data=research_data,
            analysis_result=analysis_result,
            iteration_count=iteration_count,
            error=str(exc),
        )

    task_completed = next_agent == "FINISH"

    return {
        "next_agent": next_agent,
        "task_completed": task_completed,
        "validation_notes": validation_notes,
        "messages": [
            HumanMessage(
                content=(
                    f"[Supervisor] Decision: {next_agent}. "
                    f"Reasoning: {reasoning}"
                    + (f" | Notes: {validation_notes}" if validation_notes else "")
                ),
                name="supervisor",
            )
        ],
    }


def _rule_based_decision(
    research_data: Optional[str],
    analysis_result: Optional[str],
    iteration_count: int,
    error: str = "",
) -> tuple:
    """
    Fallback rule-based decision when LLM structured output fails.

    Returns:
        Tuple of (next_agent, reasoning, validation_notes).
    """
    notes = f"LLM structured output failed ({error}); using rule-based fallback." if error else None

    if not research_data:
        return "research", "No research data yet — dispatching to Research Agent.", notes

    if not analysis_result:
        return "analyst", "Research complete but no analysis yet — dispatching to Analysis Agent.", notes

    return (
        "FINISH",
        "Both research and analysis are available — finishing pipeline.",
        notes,
    )


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_supervisor(state: OrchestratorState) -> Literal["research", "analyst", "__end__"]:
    """
    Conditional edge function that maps the supervisor's decision to a graph node.

    Args:
        state: The current OrchestratorState with next_agent set.

    Returns:
        The name of the next node ("research", "analyst") or END ("__end__").
    """
    next_agent = state.get("next_agent", "FINISH")

    if next_agent == "research":
        return "research"
    elif next_agent == "analyst":
        return "analyst"
    else:
        return END
