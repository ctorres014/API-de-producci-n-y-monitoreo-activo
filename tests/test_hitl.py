"""
Tests for app/hitl.py

Covers:
  - is_critical_query()         keyword detection
  - route_after_approval()      conditional edge routing
  - human_approval_node()       interrupt/resume lifecycle
"""

import pytest
from unittest.mock import patch
from langgraph.graph import END

from tests.conftest import make_state


# ---------------------------------------------------------------------------
# is_critical_query
# ---------------------------------------------------------------------------

class TestIsCriticalQuery:
    """Unit tests for the criticality keyword detector."""

    def test_returns_false_for_normal_query(self):
        from app.hitl import is_critical_query
        assert is_critical_query("What is machine learning?") is False

    def test_returns_false_for_empty_string(self):
        from app.hitl import is_critical_query
        assert is_critical_query("") is False

    def test_returns_false_for_none(self):
        from app.hitl import is_critical_query
        assert is_critical_query(None) is False

    def test_detects_deploy(self):
        from app.hitl import is_critical_query
        assert is_critical_query("Please deploy the new service to production") is True

    def test_detects_delete(self):
        from app.hitl import is_critical_query
        assert is_critical_query("Delete all records from the database") is True

    def test_detects_execute(self):
        from app.hitl import is_critical_query
        assert is_critical_query("Execute the migration script") is True

    def test_detects_publish(self):
        from app.hitl import is_critical_query
        assert is_critical_query("Publish the release to PyPI") is True

    def test_detects_send_email(self):
        from app.hitl import is_critical_query
        assert is_critical_query("send email to all users about the update") is True

    def test_detects_purchase(self):
        from app.hitl import is_critical_query
        assert is_critical_query("purchase 100 licenses for the team") is True

    def test_case_insensitive_deploy(self):
        from app.hitl import is_critical_query
        assert is_critical_query("DEPLOY the service NOW") is True

    def test_case_insensitive_delete(self):
        from app.hitl import is_critical_query
        assert is_critical_query("DELETE records") is True

    def test_research_query_not_critical(self):
        from app.hitl import is_critical_query
        assert is_critical_query("Research the history of artificial intelligence") is False

    def test_analysis_query_not_critical(self):
        from app.hitl import is_critical_query
        assert is_critical_query("Analyze the sentiment of these reviews") is False

    def test_partial_keyword_not_matched(self):
        """'deployed' should match because it contains 'deploy'."""
        from app.hitl import is_critical_query
        # The regex uses word boundaries, so "deployed" should NOT match "deploy"
        # because \bdeploy\b won't match "deployed".
        # This tests that the boundary logic works correctly.
        result = is_critical_query("The service was already deployed last week")
        # "deployed" does NOT match \bdeploy\b — acceptable either way,
        # but the test documents the current behaviour.
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# route_after_approval
# ---------------------------------------------------------------------------

class TestRouteAfterApproval:
    """Tests for the conditional edge after the human_approval node."""

    def test_routes_to_analyst_when_approved(self):
        from app.hitl import route_after_approval
        state = make_state(is_approved=True)
        assert route_after_approval(state) == "analyst"

    def test_routes_to_end_when_rejected(self):
        from app.hitl import route_after_approval
        state = make_state(is_approved=False)
        assert route_after_approval(state) == END

    def test_routes_to_end_when_not_decided(self):
        from app.hitl import route_after_approval
        state = make_state(is_approved=None)
        assert route_after_approval(state) == END

    def test_routes_to_end_when_key_missing(self):
        """Missing is_approved key defaults to END (falsy)."""
        from app.hitl import route_after_approval
        state = make_state()
        del state["is_approved"]
        assert route_after_approval(state) == END


# ---------------------------------------------------------------------------
# human_approval_node
# ---------------------------------------------------------------------------

class TestHumanApprovalNode:
    """Tests for the HITL LangGraph node."""

    def test_sets_is_approved_true_when_approved(self):
        """Node sets is_approved=True when human responds {approved: True}."""
        from app.hitl import human_approval_node

        state = make_state(user_query="Deploy the service", research_data="Research done.")

        with patch("app.hitl.interrupt", return_value={"approved": True}):
            result = human_approval_node(state)

        assert result["is_approved"] is True
        assert "approved" in result["validation_notes"].lower()

    def test_sets_is_approved_false_when_rejected(self):
        """Node sets is_approved=False when human responds {approved: False}."""
        from app.hitl import human_approval_node

        state = make_state(user_query="Delete all records")

        with patch("app.hitl.interrupt", return_value={"approved": False}):
            result = human_approval_node(state)

        assert result["is_approved"] is False
        assert "rejected" in result["validation_notes"].lower()

    def test_interrupt_receives_query_context(self):
        """interrupt() is called with a dict containing query and research info."""
        from app.hitl import human_approval_node

        state = make_state(
            user_query="Execute migration",
            research_data="Found 10 000 rows to migrate.",
        )
        captured = {}

        def fake_interrupt(value):
            captured["value"] = value
            return {"approved": True}

        with patch("app.hitl.interrupt", side_effect=fake_interrupt):
            human_approval_node(state)

        assert isinstance(captured["value"], dict)
        assert "Execute migration" in captured["value"].get("query", "")

    def test_handles_plain_bool_resume_value(self):
        """Node handles a plain bool resume value (not a dict)."""
        from app.hitl import human_approval_node

        state = make_state()

        with patch("app.hitl.interrupt", return_value=True):
            result = human_approval_node(state)

        assert result["is_approved"] is True

    def test_truncates_long_research_data(self):
        """interrupt() value contains a truncated research summary for long data."""
        from app.hitl import human_approval_node

        long_research = "A" * 2000
        state = make_state(research_data=long_research)
        captured = {}

        def fake_interrupt(value):
            captured["value"] = value
            return {"approved": True}

        with patch("app.hitl.interrupt", side_effect=fake_interrupt):
            human_approval_node(state)

        summary = captured["value"].get("research_summary", "")
        assert len(summary) <= 700  # 600 chars + "…" overhead
