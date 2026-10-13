"""
CLI entry point for the multi-agent orchestrator.

Provides an interactive prompt that accepts user queries, runs them through
the full Research → Analysis → Supervisor pipeline, and prints results.

Usage:
    python main.py
    python main.py --query "What is quantum computing?"
    python main.py --query "Analyze sentiment of AI news" --verbose
"""

import os
import sys
import argparse
from typing import Optional

from dotenv import load_dotenv

# Load environment variables before importing project modules
load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402
from graph import build_graph  # noqa: E402


def build_initial_state(query: str) -> dict:
    """
    Build the initial OrchestratorState for a new query.

    Args:
        query: The user's query string.

    Returns:
        A dict representing the initial state.
    """
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
    }


def run_query(query: str, verbose: bool = True) -> dict:
    """
    Run a user query through the orchestrator graph.

    Args:
        query: The user's question or task.
        verbose: If True, print intermediate progress messages.

    Returns:
        The final OrchestratorState dict after the graph finishes.
    """
    graph = build_graph()
    initial_state = build_initial_state(query)

    if verbose:
        print(f"\nProcessing query: {query!r}")
        print("-" * 60)

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        if verbose:
            print(f"Graph execution error: {exc}")
        raise

    return result


def run_query_streamed(query: str) -> dict:
    """
    Run a query and stream intermediate node outputs to stdout.

    Args:
        query: The user's question or task.

    Returns:
        The final OrchestratorState dict.
    """
    graph = build_graph()
    initial_state = build_initial_state(query)
    final_state = initial_state.copy()

    print(f"\nStreaming execution for: {query!r}")
    print("=" * 60)

    for step in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in step.items():
            print(f"\n[Node: {node_name.upper()}]")

            if node_name == "supervisor":
                next_a = node_output.get("next_agent", "?")
                notes = node_output.get("validation_notes", "")
                print(f"  Decision: {next_a}")
                if notes:
                    print(f"  Notes: {notes}")

            elif node_name == "research":
                research = node_output.get("research_data", "")
                if research:
                    preview = research[:300] + "..." if len(research) > 300 else research
                    print(f"  Research output (preview):\n  {preview}")

            elif node_name == "analyst":
                analysis = node_output.get("analysis_result", "")
                if analysis:
                    preview = analysis[:300] + "..." if len(analysis) > 300 else analysis
                    print(f"  Analysis output (preview):\n  {preview}")

            # Merge node output into final state
            if isinstance(node_output, dict):
                for k, v in node_output.items():
                    if k == "messages":
                        final_state["messages"] = final_state.get("messages", []) + (v or [])
                    elif k == "agent_contributions":
                        existing = final_state.get("agent_contributions", {})
                        final_state["agent_contributions"] = {**existing, **(v or {})}
                    elif v is not None:
                        final_state[k] = v

    return final_state


def print_results(result: dict) -> None:
    """
    Pretty-print the final orchestrator results.

    Args:
        result: The final OrchestratorState dict.
    """
    print("\n" + "=" * 60)
    print("ORCHESTRATOR RESULTS")
    print("=" * 60)

    # Agent contributions
    contributions = result.get("agent_contributions", {})
    if contributions:
        print("\nAgent Contributions:")
        for agent_name, contribution in contributions.items():
            print(f"\n  [{agent_name.upper()} AGENT]")
            preview = contribution[:600] + "..." if len(contribution) > 600 else contribution
            print(f"  {preview}")

    # Validation notes
    notes = result.get("validation_notes")
    if notes:
        print(f"\nSupervisor Notes: {notes}")

    # Summary
    research = result.get("research_data", "")
    analysis = result.get("analysis_result", "")
    iters = result.get("iteration_count", 0)
    completed = result.get("task_completed", False)

    print(f"\nTask completed: {completed}")
    print(f"Total iterations: {iters}")
    print(f"Research data length: {len(research) if research else 0} chars")
    print(f"Analysis result length: {len(analysis) if analysis else 0} chars")

    print("\n" + "=" * 60)


def interactive_loop() -> None:
    """
    Run an interactive CLI loop that accepts queries until the user exits.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    model_map = {
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "ollama": os.getenv("OLLAMA_MODEL", "llama3.2"),
    }
    model = model_map.get(provider, "unknown")

    print("\n" + "=" * 60)
    print("   Multi-Agent Orchestrator (LangGraph)")
    print("=" * 60)
    print(f"   LLM Provider : {provider}")
    print(f"   Model        : {model}")
    print(f"   Max Iterations: {os.getenv('MAX_ITERATIONS', '6')}")
    print("=" * 60)
    print("Type your query and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting. Goodbye!")
            break

        try:
            result = run_query_streamed(query)
            print_results(result)
        except Exception as exc:
            print(f"\nError processing query: {exc}")
            print("Please check your API keys and try again.\n")


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point with optional CLI argument parsing.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Orchestrator powered by LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --query 'What is quantum computing?'\n"
            "  python main.py --query 'Analyze AI regulation' --verbose\n"
        ),
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Run a single query and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose / streamed output.",
    )

    args = parser.parse_args(argv)

    if args.query:
        try:
            if args.verbose:
                result = run_query_streamed(args.query)
            else:
                result = run_query(args.query, verbose=True)
            print_results(result)
            return 0
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        interactive_loop()
        return 0


if __name__ == "__main__":
    sys.exit(main())
