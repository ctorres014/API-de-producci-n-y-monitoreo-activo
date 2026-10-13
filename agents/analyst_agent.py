"""
Analysis Agent for the multi-agent orchestrator.

Responsible for analyzing research findings using quantitative
and qualitative tools. Outputs analysis results into the shared state.
"""

from typing import Any

from langchain_core.messages import HumanMessage, AIMessage

from state import OrchestratorState


ANALYST_SYSTEM_PROMPT = """You are a specialized Analysis Agent in a multi-agent system.

Your responsibilities:
1. Analyze data and text provided by the Research Agent
2. Perform sentiment analysis when dealing with opinions, reviews, or subjective content
3. Execute mathematical calculations when numerical analysis is needed
4. Validate data structures and schemas when working with structured data
5. Summarize complex information into actionable insights

Guidelines:
- Use your analysis tools for quantitative and qualitative analysis
- Provide concrete numbers and metrics when possible
- Identify patterns, trends, and key insights
- Present findings in a structured, clear format
- Base your analysis on the research data provided

Your analysis will be reviewed by the Supervisor for completeness."""


def create_analyst_agent(llm: Any = None) -> Any:
    """
    Create and return an Analysis Agent using LangGraph's prebuilt ReAct agent.

    Args:
        llm: An optional LangChain chat model. If None, the default from config is used.

    Returns:
        A compiled LangGraph ReAct agent.
    """
    import warnings
    from config import get_llm
    from tools.analysis_tools import (
        sentiment_analyzer,
        math_calculator,
        data_validator,
        text_summarizer,
    )

    # Silence the LangGraph 1.x migration warning — create_react_agent still
    # lives in langgraph.prebuilt; the suggested target (langchain.agents) does
    # not export it yet in this version.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from langgraph.prebuilt import create_react_agent

    if llm is None:
        llm = get_llm()

    tools = [sentiment_analyzer, math_calculator, data_validator, text_summarizer]
    return create_react_agent(llm, tools, prompt=ANALYST_SYSTEM_PROMPT)


def analyst_node(state: OrchestratorState) -> dict:
    """
    LangGraph node that runs the Analysis Agent and updates shared state.

    Passes research_data and the user query to the analyst agent,
    then merges the analysis output into analysis_result and agent_contributions.

    Args:
        state: The current OrchestratorState.

    Returns:
        A partial state dict with updated messages, analysis_result,
        agent_contributions, and iteration_count.
    """
    try:
        user_query = state.get("user_query", "")
        research_data = state.get("research_data", "")
        iteration_count = state.get("iteration_count", 0)
        existing_contributions = state.get("agent_contributions", {})

        # Build context-rich message for the analyst
        if research_data:
            analysis_input_content = (
                f"Research findings:\n{research_data}\n\n"
                f"User query: {user_query}\n\n"
                f"Please analyze these findings."
            )
        else:
            analysis_input_content = (
                f"User query: {user_query}\n\n"
                f"No prior research data is available. "
                f"Please analyze the query directly using your tools."
            )

        agent = create_analyst_agent()
        agent_input = {"messages": [HumanMessage(content=analysis_input_content)]}

        result = agent.invoke(agent_input)

        # Extract the final AI response
        agent_messages = result.get("messages", [])
        analysis_output = ""
        for msg in reversed(agent_messages):
            if isinstance(msg, AIMessage) and msg.content:
                analysis_output = msg.content
                break

        if not analysis_output:
            analysis_output = (
                f"Analysis completed for query: '{user_query}'. "
                "No detailed analysis was generated."
            )

        # Merge agent contributions (additive)
        updated_contributions = {**existing_contributions, "analyst": analysis_output}

        return {
            "messages": [
                AIMessage(
                    content=f"[Analysis Agent] {analysis_output}",
                    name="analyst_agent",
                )
            ],
            "analysis_result": analysis_output,
            "agent_contributions": updated_contributions,
            "iteration_count": iteration_count + 1,
        }

    except Exception as exc:
        error_message = (
            f"Analysis agent encountered an error: {exc}. "
            "Continuing with available information."
        )
        existing_contributions = state.get("agent_contributions", {})
        return {
            "messages": [AIMessage(content=f"[Analysis Agent Error] {error_message}", name="analyst_agent")],
            "analysis_result": state.get("analysis_result") or error_message,
            "agent_contributions": {**existing_contributions, "analyst": f"Error: {exc}"},
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
