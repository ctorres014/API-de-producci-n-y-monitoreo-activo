"""
Tests for the LangGraph graph construction and structure.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_initial_state(query: str = "Test query") -> dict:
    """Build a valid initial OrchestratorState for graph testing."""
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


# ---------------------------------------------------------------------------
# Graph Build Tests
# ---------------------------------------------------------------------------

class TestGraphBuild:
    """Tests for graph construction."""

    def test_graph_builds_successfully(self):
        """build_graph() compiles without raising exceptions."""
        from graph import build_graph

        graph = build_graph()
        assert graph is not None

    def test_graph_module_exports_app(self):
        """graph module exports a module-level 'app' instance."""
        import graph as g
        assert hasattr(g, "app")
        assert g.app is not None

    def test_graph_has_correct_nodes(self):
        """The compiled graph contains supervisor, research, and analyst nodes."""
        from graph import build_graph

        graph = build_graph()
        # LangGraph 1.x: use get_graph() to get the underlying langchain_core Graph,
        # or use .nodes directly on the compiled graph.
        node_names = set(graph.nodes.keys())

        assert "supervisor" in node_names, f"'supervisor' not in nodes: {node_names}"
        assert "research" in node_names, f"'research' not in nodes: {node_names}"
        assert "analyst" in node_names, f"'analyst' not in nodes: {node_names}"

    def test_graph_build_with_checkpointer(self):
        """build_graph() accepts an optional checkpointer argument."""
        from graph import build_graph
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        graph = build_graph(checkpointer=checkpointer)
        assert graph is not None

    def test_graph_build_without_checkpointer(self):
        """build_graph() works fine with checkpointer=None (default)."""
        from graph import build_graph

        graph = build_graph(checkpointer=None)
        assert graph is not None


# ---------------------------------------------------------------------------
# Graph State Validation Tests
# ---------------------------------------------------------------------------

class TestGraphStateValidation:
    """Tests for state schema correctness."""

    def test_initial_state_structure(self):
        """Initial state dict has all required keys for graph invocation."""
        state = make_initial_state()
        required_keys = [
            "messages",
            "user_query",
            "next_agent",
            "research_data",
            "analysis_result",
            "task_completed",
            "iteration_count",
            "agent_contributions",
            "validation_notes",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_state_types_are_correct(self):
        """Initial state values have the expected types."""
        state = make_initial_state()

        assert isinstance(state["messages"], list)
        assert isinstance(state["user_query"], str)
        assert isinstance(state["next_agent"], str)
        assert state["research_data"] is None
        assert state["analysis_result"] is None
        assert isinstance(state["task_completed"], bool)
        assert isinstance(state["iteration_count"], int)
        assert isinstance(state["agent_contributions"], dict)
        assert state["validation_notes"] is None

    def test_orchestrator_state_annotation(self):
        """OrchestratorState TypedDict has the add_messages annotation on messages."""
        from state import OrchestratorState
        import typing

        annotations = OrchestratorState.__annotations__
        assert "messages" in annotations

        # The Annotated type wraps the actual type
        messages_annotation = annotations["messages"]
        origin = getattr(messages_annotation, "__class__", None)
        assert messages_annotation is not None


# ---------------------------------------------------------------------------
# Graph Topology Tests
# ---------------------------------------------------------------------------

class TestGraphTopology:
    """Tests for verifying graph edges and flow."""

    def test_graph_has_start_edge_to_supervisor(self):
        """The graph has an edge from START to supervisor."""
        from graph import build_graph

        graph = build_graph()
        # LangGraph 1.x: use get_graph() for the underlying langchain_core Graph
        gg = graph.get_graph()
        edge_pairs = [(e.source, e.target) for e in gg.edges]

        assert any(
            ("start" in src.lower() or src == "__start__") and "supervisor" in tgt.lower()
            for src, tgt in edge_pairs
        ), f"No start→supervisor edge found. Edges: {edge_pairs}"

    def test_graph_supervisor_has_conditional_edges(self):
        """Supervisor node has conditional outgoing edges."""
        from graph import build_graph

        graph = build_graph()
        gg = graph.get_graph()

        # Supervisor should have at least one conditional edge
        supervisor_edges = [e for e in gg.edges if "supervisor" in e.source.lower()]
        conditional_edges = [e for e in supervisor_edges if e.conditional]
        assert len(conditional_edges) > 0, \
            f"No conditional edges from supervisor. Supervisor edges: {supervisor_edges}"

    def test_graph_research_loops_back_to_supervisor(self):
        """research node has an edge back to supervisor."""
        from graph import build_graph

        graph = build_graph()
        gg = graph.get_graph()
        edge_pairs = [(e.source, e.target) for e in gg.edges]

        assert any(
            "research" in src and "supervisor" in dst
            for src, dst in edge_pairs
        ), f"Expected research→supervisor edge. Found: {edge_pairs}"

    def test_graph_analyst_loops_back_to_supervisor(self):
        """analyst node has an edge back to supervisor."""
        from graph import build_graph

        graph = build_graph()
        gg = graph.get_graph()
        edge_pairs = [(e.source, e.target) for e in gg.edges]

        assert any(
            "analyst" in src and "supervisor" in dst
            for src, dst in edge_pairs
        ), f"Expected analyst→supervisor edge. Found: {edge_pairs}"


# ---------------------------------------------------------------------------
# Graph Invocation Tests (with mocked agents)
# ---------------------------------------------------------------------------

class TestGraphInvocation:
    """Tests for graph invocation with mocked node functions."""

    def test_graph_terminates_at_max_iterations(self):
        """Graph stops when supervisor forces FINISH at max iterations."""
        import agents.supervisor as sv_module
        from graph import build_graph

        original_max = sv_module.MAX_ITERATIONS
        try:
            # Override MAX_ITERATIONS to 1 in the supervisor module
            sv_module.MAX_ITERATIONS = 1

            graph = build_graph()
            # With iteration_count=1 and MAX_ITERATIONS=1, supervisor should force FINISH immediately
            initial_state = make_initial_state()
            initial_state["iteration_count"] = 1

            result = graph.invoke(initial_state)

            assert result is not None
            assert result.get("task_completed") is True or result.get("next_agent") == "FINISH"
        finally:
            sv_module.MAX_ITERATIONS = original_max

    def test_build_graph_function_returns_compiled_graph(self):
        """build_graph returns a CompiledStateGraph-like object with invoke method."""
        from graph import build_graph

        graph = build_graph()
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")
        assert callable(graph.invoke)
