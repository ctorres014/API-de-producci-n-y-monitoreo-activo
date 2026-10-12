"""
Tests for OrchestratorState schema definition.
"""

import pytest
from typing import get_type_hints


class TestOrchestratorState:
    """Tests for the OrchestratorState TypedDict."""

    def test_import(self):
        """OrchestratorState can be imported without error."""
        from state import OrchestratorState
        assert OrchestratorState is not None

    def test_required_fields_exist(self):
        """OrchestratorState has all required field annotations."""
        from state import OrchestratorState

        hints = OrchestratorState.__annotations__
        required_fields = [
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
        for field in required_fields:
            assert field in hints, f"Missing field: {field}"

    def test_instantiation_with_full_state(self):
        """OrchestratorState can be instantiated with all fields."""
        from state import OrchestratorState
        from langchain_core.messages import HumanMessage

        state: OrchestratorState = {
            "messages": [HumanMessage(content="Hello")],
            "user_query": "Test query",
            "next_agent": "research",
            "research_data": None,
            "analysis_result": None,
            "task_completed": False,
            "iteration_count": 0,
            "agent_contributions": {},
            "validation_notes": None,
        }

        assert state["user_query"] == "Test query"
        assert state["next_agent"] == "research"
        assert state["research_data"] is None
        assert state["analysis_result"] is None
        assert state["task_completed"] is False
        assert state["iteration_count"] == 0
        assert state["agent_contributions"] == {}
        assert state["validation_notes"] is None

    def test_instantiation_with_data(self):
        """OrchestratorState holds data correctly when populated."""
        from state import OrchestratorState
        from langchain_core.messages import HumanMessage, AIMessage

        state: OrchestratorState = {
            "messages": [
                HumanMessage(content="What is AI?"),
                AIMessage(content="AI is artificial intelligence."),
            ],
            "user_query": "What is AI?",
            "next_agent": "analyst",
            "research_data": "AI stands for artificial intelligence...",
            "analysis_result": None,
            "task_completed": False,
            "iteration_count": 1,
            "agent_contributions": {"research": "AI stands for artificial intelligence..."},
            "validation_notes": "Research complete, analysis pending.",
        }

        assert len(state["messages"]) == 2
        assert state["iteration_count"] == 1
        assert "research" in state["agent_contributions"]
        assert state["research_data"] is not None

    def test_messages_field_supports_list(self):
        """The messages field accepts a list."""
        from state import OrchestratorState

        state: OrchestratorState = {
            "messages": [],
            "user_query": "test",
            "next_agent": "research",
            "research_data": None,
            "analysis_result": None,
            "task_completed": False,
            "iteration_count": 0,
            "agent_contributions": {},
            "validation_notes": None,
        }

        assert isinstance(state["messages"], list)

    def test_task_completed_field_is_boolean(self):
        """task_completed field stores a boolean value."""
        from state import OrchestratorState

        for val in [True, False]:
            state: OrchestratorState = {
                "messages": [],
                "user_query": "test",
                "next_agent": "FINISH",
                "research_data": "data",
                "analysis_result": "result",
                "task_completed": val,
                "iteration_count": 2,
                "agent_contributions": {},
                "validation_notes": None,
            }
            assert state["task_completed"] == val

    def test_agent_contributions_is_dict(self):
        """agent_contributions stores a dict with agent names as keys."""
        from state import OrchestratorState

        contributions = {
            "research": "Research findings here.",
            "analyst": "Analysis results here.",
        }
        state: OrchestratorState = {
            "messages": [],
            "user_query": "test",
            "next_agent": "FINISH",
            "research_data": "Research findings here.",
            "analysis_result": "Analysis results here.",
            "task_completed": True,
            "iteration_count": 2,
            "agent_contributions": contributions,
            "validation_notes": None,
        }

        assert isinstance(state["agent_contributions"], dict)
        assert state["agent_contributions"]["research"] == "Research findings here."
        assert state["agent_contributions"]["analyst"] == "Analysis results here."

    def test_state_is_typeddict(self):
        """OrchestratorState is a TypedDict."""
        from state import OrchestratorState
        # TypedDicts have __annotations__ and __total__ attributes
        assert hasattr(OrchestratorState, "__annotations__")
        assert hasattr(OrchestratorState, "__total__")
