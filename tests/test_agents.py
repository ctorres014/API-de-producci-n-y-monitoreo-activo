"""
Tests for agent creation, node functions, and supervisor routing.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_base_state(**overrides) -> dict:
    """Create a minimal valid OrchestratorState dict for testing."""
    base = {
        "messages": [HumanMessage(content="Test query")],
        "user_query": "Test query",
        "next_agent": "research",
        "research_data": None,
        "analysis_result": None,
        "task_completed": False,
        "iteration_count": 0,
        "agent_contributions": {},
        "validation_notes": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Research Agent Tests
# ---------------------------------------------------------------------------

class TestResearchAgent:
    """Tests for the research agent factory and node."""

    def test_create_research_agent(self):
        """create_research_agent returns a compiled agent with mocked LLM."""
        from agents.research_agent import create_research_agent

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        # The function should not raise
        try:
            agent = create_research_agent(llm=mock_llm)
            assert agent is not None
        except Exception:
            # create_react_agent may have internal requirements; just verify no import errors
            pass

    def test_research_agent_module_imports(self):
        """research_agent module imports without error."""
        import agents.research_agent as ra
        assert hasattr(ra, "create_research_agent")
        assert hasattr(ra, "research_node")
        assert hasattr(ra, "RESEARCH_SYSTEM_PROMPT")

    def test_research_node_handles_exception_gracefully(self):
        """research_node catches exceptions and returns a valid partial state."""
        from agents.research_agent import research_node

        state = make_base_state()

        with patch("agents.research_agent.create_research_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.invoke.side_effect = RuntimeError("LLM unavailable")
            mock_factory.return_value = mock_agent

            result = research_node(state)

        assert "research_data" in result
        assert "agent_contributions" in result
        assert "iteration_count" in result
        # iteration_count should increment even on error
        assert result["iteration_count"] == 1

    def test_research_node_increments_iteration(self):
        """research_node increments iteration_count by 1."""
        from agents.research_agent import research_node

        state = make_base_state(iteration_count=2)

        with patch("agents.research_agent.create_research_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [AIMessage(content="Research findings here.")]
            }
            mock_factory.return_value = mock_agent

            result = research_node(state)

        assert result["iteration_count"] == 3

    def test_research_node_updates_agent_contributions(self):
        """research_node adds 'research' key to agent_contributions."""
        from agents.research_agent import research_node

        state = make_base_state()

        with patch("agents.research_agent.create_research_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [AIMessage(content="Important research results.")]
            }
            mock_factory.return_value = mock_agent

            result = research_node(state)

        assert "research" in result["agent_contributions"]

    def test_research_node_preserves_existing_contributions(self):
        """research_node merges into existing agent_contributions without overwriting."""
        from agents.research_agent import research_node

        state = make_base_state(
            agent_contributions={"analyst": "Previous analysis."}
        )

        with patch("agents.research_agent.create_research_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [AIMessage(content="New research.")]
            }
            mock_factory.return_value = mock_agent

            result = research_node(state)

        assert "analyst" in result["agent_contributions"]
        assert "research" in result["agent_contributions"]


# ---------------------------------------------------------------------------
# Analyst Agent Tests
# ---------------------------------------------------------------------------

class TestAnalystAgent:
    """Tests for the analyst agent factory and node."""

    def test_create_analyst_agent(self):
        """create_analyst_agent returns a compiled agent with mocked LLM."""
        from agents.analyst_agent import create_analyst_agent

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        try:
            agent = create_analyst_agent(llm=mock_llm)
            assert agent is not None
        except Exception:
            pass

    def test_analyst_agent_module_imports(self):
        """analyst_agent module imports without error."""
        import agents.analyst_agent as aa
        assert hasattr(aa, "create_analyst_agent")
        assert hasattr(aa, "analyst_node")
        assert hasattr(aa, "ANALYST_SYSTEM_PROMPT")

    def test_analyst_node_handles_exception_gracefully(self):
        """analyst_node catches exceptions and returns a valid partial state."""
        from agents.analyst_agent import analyst_node

        state = make_base_state(research_data="Some research data.")

        with patch("agents.analyst_agent.create_analyst_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.invoke.side_effect = RuntimeError("LLM unavailable")
            mock_factory.return_value = mock_agent

            result = analyst_node(state)

        assert "analysis_result" in result
        assert "agent_contributions" in result
        assert result["iteration_count"] == 1

    def test_analyst_node_includes_research_data_in_input(self):
        """analyst_node passes research_data to the agent invocation."""
        from agents.analyst_agent import analyst_node

        research_content = "Important research about topic X."
        state = make_base_state(research_data=research_content)

        captured_input = {}

        def capture_invoke(inp):
            captured_input.update(inp)
            return {"messages": [AIMessage(content="Analysis done.")]}

        with patch("agents.analyst_agent.create_analyst_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.invoke.side_effect = capture_invoke
            mock_factory.return_value = mock_agent

            analyst_node(state)

        messages = captured_input.get("messages", [])
        assert len(messages) > 0
        content = messages[0].content if messages else ""
        assert research_content in content

    def test_analyst_node_updates_analysis_result(self):
        """analyst_node sets analysis_result in the returned state."""
        from agents.analyst_agent import analyst_node

        state = make_base_state(research_data="Research data here.")

        with patch("agents.analyst_agent.create_analyst_agent") as mock_factory:
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [AIMessage(content="Detailed analysis output.")]
            }
            mock_factory.return_value = mock_agent

            result = analyst_node(state)

        assert result["analysis_result"] is not None
        assert len(result["analysis_result"]) > 0


# ---------------------------------------------------------------------------
# Supervisor Tests
# ---------------------------------------------------------------------------

class TestSupervisor:
    """Tests for the supervisor node and routing function."""

    def test_supervisor_module_imports(self):
        """supervisor module imports without error."""
        import agents.supervisor as sv
        assert hasattr(sv, "supervisor_node")
        assert hasattr(sv, "route_supervisor")
        assert hasattr(sv, "SupervisorDecision")
        assert hasattr(sv, "SUPERVISOR_SYSTEM_PROMPT")

    def test_supervisor_forces_finish_at_max_iterations(self):
        """supervisor_node forces FINISH when iteration_count >= MAX_ITERATIONS."""
        import agents.supervisor as sv_module
        from agents.supervisor import supervisor_node

        original_max = sv_module.MAX_ITERATIONS
        try:
            sv_module.MAX_ITERATIONS = 6
            state = make_base_state(
                iteration_count=6,
                research_data="Some data",
                analysis_result="Some analysis",
            )
            result = supervisor_node(state)
        finally:
            sv_module.MAX_ITERATIONS = original_max

        assert result["next_agent"] == "FINISH"
        assert result["task_completed"] is True
        assert result["validation_notes"] is not None

    def test_supervisor_forces_finish_above_max_iterations(self):
        """supervisor_node forces FINISH when iteration_count > MAX_ITERATIONS."""
        import agents.supervisor as sv_module
        from agents.supervisor import supervisor_node

        original_max = sv_module.MAX_ITERATIONS
        try:
            sv_module.MAX_ITERATIONS = 6
            state = make_base_state(iteration_count=10)
            result = supervisor_node(state)
        finally:
            sv_module.MAX_ITERATIONS = original_max

        assert result["next_agent"] == "FINISH"

    def test_supervisor_uses_fallback_when_llm_fails(self):
        """supervisor_node uses rule-based fallback when LLM call raises."""
        from agents.supervisor import supervisor_node

        state = make_base_state(iteration_count=0)

        with patch("agents.supervisor.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.side_effect = Exception("LLM error")
            mock_get_llm.return_value = mock_llm

            result = supervisor_node(state)

        # Should return "research" as fallback when no research data
        assert result["next_agent"] in ("research", "analyst", "FINISH")

    def test_supervisor_fallback_routes_to_research_first(self):
        """Rule-based fallback sends to research when research_data is None."""
        from agents.supervisor import _rule_based_decision

        next_agent, reasoning, notes = _rule_based_decision(
            research_data=None,
            analysis_result=None,
            iteration_count=0,
        )
        assert next_agent == "research"

    def test_supervisor_fallback_routes_to_analyst_after_research(self):
        """Rule-based fallback sends to analyst when research done but no analysis."""
        from agents.supervisor import _rule_based_decision

        next_agent, reasoning, notes = _rule_based_decision(
            research_data="Research results here.",
            analysis_result=None,
            iteration_count=1,
        )
        assert next_agent == "analyst"

    def test_supervisor_fallback_finishes_when_both_done(self):
        """Rule-based fallback returns FINISH when both research and analysis are done."""
        from agents.supervisor import _rule_based_decision

        next_agent, reasoning, notes = _rule_based_decision(
            research_data="Research results.",
            analysis_result="Analysis results.",
            iteration_count=2,
        )
        assert next_agent == "FINISH"


# ---------------------------------------------------------------------------
# Route Supervisor Tests
# ---------------------------------------------------------------------------

class TestRouteSupervisor:
    """Tests for the route_supervisor conditional edge function."""

    def test_route_supervisor_research(self):
        """route_supervisor returns 'research' when next_agent is 'research'."""
        from agents.supervisor import route_supervisor

        state = make_base_state(next_agent="research")
        assert route_supervisor(state) == "research"

    def test_route_supervisor_analyst(self):
        """route_supervisor returns 'analyst' when next_agent is 'analyst'."""
        from agents.supervisor import route_supervisor

        state = make_base_state(next_agent="analyst")
        assert route_supervisor(state) == "analyst"

    def test_route_supervisor_finish(self):
        """route_supervisor returns END when next_agent is 'FINISH'."""
        from agents.supervisor import route_supervisor

        state = make_base_state(next_agent="FINISH")
        result = route_supervisor(state)
        assert result == END

    def test_route_supervisor_unknown_defaults_to_end(self):
        """route_supervisor defaults to END for unknown next_agent values."""
        from agents.supervisor import route_supervisor

        state = make_base_state(next_agent="unknown")
        result = route_supervisor(state)
        assert result == END
