"""
Research Agent for the multi-agent orchestrator.

Responsible for gathering information using web search and Wikipedia tools.
Outputs research findings into the shared OrchestratorState.
"""

from typing import Any

from langchain_core.messages import HumanMessage, AIMessage

from state import OrchestratorState


RESEARCH_SYSTEM_PROMPT = """You are a specialized Research Agent in a multi-agent system.

Your responsibilities:
1. Search for relevant information using available tools
2. Gather facts, data, and context about the requested topic
3. Synthesize findings into a clear, structured summary

Guidelines:
- Always use your search tools to find current information
- Cite sources when available
- Structure your output clearly with key findings
- Focus on factual, verifiable information
- After gathering information, provide a comprehensive research summary

Your output will be passed to an Analysis Agent for further processing."""


def create_research_agent(llm: Any = None) -> Any:
    """
    Create and return a Research Agent using LangGraph's prebuilt ReAct agent.

    Args:
        llm: An optional LangChain chat model. If None, the default from config is used.

    Returns:
        A compiled LangGraph ReAct agent.
    """
    import warnings
    from config import get_llm
    from tools.search_tools import web_search, wikipedia_search

    # Silence the LangGraph 1.x migration warning — create_react_agent still
    # lives in langgraph.prebuilt; the suggested target (langchain.agents) does
    # not export it yet in this version.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from langgraph.prebuilt import create_react_agent

    if llm is None:
        llm = get_llm()

    tools = [web_search, wikipedia_search]
    return create_react_agent(llm, tools, prompt=RESEARCH_SYSTEM_PROMPT)


def research_node(state: OrchestratorState) -> dict:
    """
    LangGraph node that runs the Research Agent and updates shared state.

    Extracts the user query from state, calls the research agent,
    and merges the research output into research_data and agent_contributions.

    Args:
        state: The current OrchestratorState.

    Returns:
        A partial state dict with updated messages, research_data,
        agent_contributions, and iteration_count.
    """
    try:
        user_query = state.get("user_query", "")
        messages = state.get("messages", [])
        iteration_count = state.get("iteration_count", 0)
        existing_contributions = state.get("agent_contributions", {})

        # Build the input message for the research agent
        # Include any prior instructions from the supervisor in messages
        research_input_content = f"Please research the following topic:\n\n{user_query}"

        # Check if the supervisor added any specific instructions
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                content_lower = msg.content.lower()
                if "research" in content_lower and "focus" in content_lower:
                    research_input_content = (
                        f"{msg.content}\n\nOriginal query: {user_query}"
                    )
                    break

        agent = create_research_agent()
        agent_input = {"messages": [HumanMessage(content=research_input_content)]}

        result = agent.invoke(agent_input)

        # Extract the final AI response
        agent_messages = result.get("messages", [])
        research_output = ""
        for msg in reversed(agent_messages):
            if isinstance(msg, AIMessage) and msg.content:
                research_output = msg.content
                break

        if not research_output:
            research_output = (
                f"Research completed for query: '{user_query}'. "
                "No detailed findings were generated."
            )

        # Merge agent contributions (additive)
        updated_contributions = {**existing_contributions, "research": research_output}

        return {
            "messages": [
                AIMessage(
                    content=f"[Research Agent] {research_output}",
                    name="research_agent",
                )
            ],
            "research_data": research_output,
            "agent_contributions": updated_contributions,
            "iteration_count": iteration_count + 1,
        }

    except Exception as exc:
        error_message = (
            f"Research agent encountered an error: {exc}. "
            "Continuing with available information."
        )
        existing_contributions = state.get("agent_contributions", {})
        return {
            "messages": [AIMessage(content=f"[Research Agent Error] {error_message}", name="research_agent")],
            "research_data": state.get("research_data") or error_message,
            "agent_contributions": {**existing_contributions, "research": f"Error: {exc}"},
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
