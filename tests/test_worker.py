"""
Tests for app/worker.py

Covers:
  - create_job()       job creation and Redis persistence
  - get_job()          retrieval and missing-key handling
  - update_job()       field-level updates and merging
  - run_graph()        success, failure, and interrupt (HITL) paths
  - resume_graph()     approved, rejected, and exception paths
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import FakeRedis, make_state
from app.worker import (
    create_job,
    get_job,
    update_job,
    run_graph,
    resume_graph,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_AWAITING,
    STATUS_REJECTED,
    _job_key,
)


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------

class TestCreateJob:

    def test_returns_string_job_id(self, fake_redis):
        job_id = create_job(fake_redis, "What is AI?")
        assert isinstance(job_id, str)
        assert len(job_id) > 0

    def test_job_id_is_unique(self, fake_redis):
        ids = {create_job(fake_redis, "query") for _ in range(10)}
        assert len(ids) == 10

    def test_job_stored_in_redis(self, fake_redis):
        job_id = create_job(fake_redis, "Test query")
        stored = fake_redis.get_parsed(_job_key(job_id))
        assert stored is not None
        assert stored["job_id"] == job_id

    def test_initial_status_is_pending(self, fake_redis):
        job_id = create_job(fake_redis, "Some query")
        stored = fake_redis.get_parsed(_job_key(job_id))
        assert stored["status"] == STATUS_PENDING

    def test_query_is_stored(self, fake_redis):
        query = "Explain transformers"
        job_id = create_job(fake_redis, query)
        stored = fake_redis.get_parsed(_job_key(job_id))
        assert stored["query"] == query

    def test_result_and_error_are_null_initially(self, fake_redis):
        job_id = create_job(fake_redis, "q")
        stored = fake_redis.get_parsed(_job_key(job_id))
        assert stored["result"] is None
        assert stored["error"] is None

    def test_created_at_and_updated_at_are_set(self, fake_redis):
        job_id = create_job(fake_redis, "q")
        stored = fake_redis.get_parsed(_job_key(job_id))
        assert stored["created_at"] is not None
        assert stored["updated_at"] is not None


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------

class TestGetJob:

    def test_returns_none_for_missing_job(self, fake_redis):
        assert get_job(fake_redis, "nonexistent-id") is None

    def test_returns_dict_for_existing_job(self, fake_redis):
        job_id = create_job(fake_redis, "query")
        result = get_job(fake_redis, job_id)
        assert isinstance(result, dict)
        assert result["job_id"] == job_id

    def test_returned_dict_has_all_required_keys(self, fake_redis):
        job_id = create_job(fake_redis, "query")
        job = get_job(fake_redis, job_id)
        for key in ("job_id", "status", "query", "result", "error", "created_at", "updated_at"):
            assert key in job, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# update_job
# ---------------------------------------------------------------------------

class TestUpdateJob:

    def test_updates_status(self, fake_redis):
        job_id = create_job(fake_redis, "query")
        update_job(fake_redis, job_id, status=STATUS_RUNNING)
        job = get_job(fake_redis, job_id)
        assert job["status"] == STATUS_RUNNING

    def test_updates_multiple_fields(self, fake_redis):
        job_id = create_job(fake_redis, "query")
        update_job(fake_redis, job_id, status=STATUS_DONE, result={"answer": "yes"})
        job = get_job(fake_redis, job_id)
        assert job["status"] == STATUS_DONE
        assert job["result"] == {"answer": "yes"}

    def test_preserves_unmodified_fields(self, fake_redis):
        query = "original query"
        job_id = create_job(fake_redis, query)
        update_job(fake_redis, job_id, status=STATUS_RUNNING)
        job = get_job(fake_redis, job_id)
        assert job["query"] == query

    def test_update_refreshes_updated_at(self, fake_redis):
        job_id = create_job(fake_redis, "q")
        original = get_job(fake_redis, job_id)["updated_at"]
        import time; time.sleep(0.01)
        update_job(fake_redis, job_id, status=STATUS_RUNNING)
        refreshed = get_job(fake_redis, job_id)["updated_at"]
        assert refreshed >= original

    def test_update_on_missing_job_does_not_raise(self, fake_redis):
        """update_job on a nonexistent job logs an error but doesn't raise."""
        update_job(fake_redis, "ghost-id", status=STATUS_FAILED)  # should not raise


# ---------------------------------------------------------------------------
# run_graph — success path
# ---------------------------------------------------------------------------

class TestRunGraphSuccess:

    def test_sets_status_done_on_success(self, fake_redis):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = make_state(
            task_completed=True, research_data="data", analysis_result="result"
        )
        job_id = create_job(fake_redis, "What is AI?")

        run_graph(fake_redis, mock_graph, job_id, "What is AI?")

        job = get_job(fake_redis, job_id)
        assert job["status"] == STATUS_DONE

    def test_stores_result_on_success(self, fake_redis):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = make_state(
            research_data="Research output",
            analysis_result="Analysis output",
            task_completed=True,
        )
        job_id = create_job(fake_redis, "query")

        run_graph(fake_redis, mock_graph, job_id, "query")

        job = get_job(fake_redis, job_id)
        assert job["result"] is not None
        assert job["result"]["research_data"] == "Research output"
        assert job["result"]["analysis_result"] == "Analysis output"

    def test_sets_running_before_invoke(self, fake_redis):
        """Status should transition to RUNNING before graph.invoke() is called."""
        observed_statuses = []

        def capture_then_return(initial_state, config):
            observed_statuses.append(get_job(fake_redis, job_id)["status"])
            return make_state(task_completed=True)

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = capture_then_return

        job_id = create_job(fake_redis, "q")
        run_graph(fake_redis, mock_graph, job_id, "q")

        assert STATUS_RUNNING in observed_statuses


# ---------------------------------------------------------------------------
# run_graph — failure path
# ---------------------------------------------------------------------------

class TestRunGraphFailure:

    def test_sets_status_failed_on_exception(self, fake_redis):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = ValueError("LLM unavailable")

        job_id = create_job(fake_redis, "query")
        run_graph(fake_redis, mock_graph, job_id, "query")

        job = get_job(fake_redis, job_id)
        assert job["status"] == STATUS_FAILED

    def test_stores_error_message_on_failure(self, fake_redis):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("Connection timeout")

        job_id = create_job(fake_redis, "query")
        run_graph(fake_redis, mock_graph, job_id, "query")

        job = get_job(fake_redis, job_id)
        assert "Connection timeout" in (job.get("error") or "")


# ---------------------------------------------------------------------------
# run_graph — HITL interrupt path
# ---------------------------------------------------------------------------

class TestRunGraphInterrupt:

    def _make_interrupt_exception(self, value=None):
        """Create an exception that looks like GraphInterrupt to the worker."""
        class FakeGraphInterrupt(Exception):
            pass
        FakeGraphInterrupt.__name__ = "GraphInterrupt"

        exc = FakeGraphInterrupt("interrupted")
        # Simulate the structure LangGraph uses: exc.args[0] is a list of Interrupt objects
        class FakeInterrupt:
            def __init__(self, v):
                self.value = v
        exc.args = ([FakeInterrupt(value or {"message": "needs approval"})],)
        return exc

    def test_sets_status_awaiting_on_interrupt(self, fake_redis):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = self._make_interrupt_exception()

        job_id = create_job(fake_redis, "deploy service")
        run_graph(fake_redis, mock_graph, job_id, "deploy service")

        job = get_job(fake_redis, job_id)
        assert job["status"] == STATUS_AWAITING

    def test_stores_interrupt_value(self, fake_redis):
        interrupt_payload = {"message": "approval needed", "query": "deploy service"}
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = self._make_interrupt_exception(interrupt_payload)

        job_id = create_job(fake_redis, "deploy service")
        run_graph(fake_redis, mock_graph, job_id, "deploy service")

        job = get_job(fake_redis, job_id)
        assert job["interrupt_value"] is not None

    def test_sets_status_awaiting_when_invoke_returns_interrupt(self, fake_redis):
        """LangGraph >= 0.4 returns pending interrupts under '__interrupt__' instead of raising."""
        class FakeInterrupt:
            def __init__(self, v):
                self.value = v

        payload = {"message": "approval needed", "query": "deploy service"}
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            **make_state(research_data="data"),
            "__interrupt__": [FakeInterrupt(payload)],
        }

        job_id = create_job(fake_redis, "deploy service")
        run_graph(fake_redis, mock_graph, job_id, "deploy service")

        job = get_job(fake_redis, job_id)
        assert job["status"] == STATUS_AWAITING
        assert job["interrupt_value"] == payload
        assert job["result"] is None


# ---------------------------------------------------------------------------
# resume_graph
# ---------------------------------------------------------------------------

class TestResumeGraph:

    def test_resumes_with_approved_true(self, fake_redis):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = make_state(task_completed=True)

        job_id = create_job(fake_redis, "deploy")
        with patch("langgraph.types.Command") as mock_cmd:
            mock_cmd.return_value = object()
            resume_graph(fake_redis, mock_graph, job_id, approved=True)

        job = get_job(fake_redis, job_id)
        assert job["status"] == STATUS_DONE

    def test_resumes_with_approved_false_sets_rejected(self, fake_redis):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = make_state(task_completed=False)

        job_id = create_job(fake_redis, "deploy")
        with patch("langgraph.types.Command") as mock_cmd:
            mock_cmd.return_value = object()
            resume_graph(fake_redis, mock_graph, job_id, approved=False)

        job = get_job(fake_redis, job_id)
        assert job["status"] == STATUS_REJECTED

    def test_sets_failed_on_exception_during_resume(self, fake_redis):
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("Graph error during resume")

        job_id = create_job(fake_redis, "deploy")
        with patch("langgraph.types.Command"):
            resume_graph(fake_redis, mock_graph, job_id, approved=True)

        job = get_job(fake_redis, job_id)
        assert job["status"] == STATUS_FAILED

    def test_calls_graph_invoke_with_command(self, fake_redis):
        """resume_graph passes a Command object to graph.invoke, not a plain dict."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = make_state()

        job_id = create_job(fake_redis, "query")
        sentinel = object()

        with patch("langgraph.types.Command", return_value=sentinel):
            resume_graph(fake_redis, mock_graph, job_id, approved=True)

        call_args = mock_graph.invoke.call_args
        assert call_args[0][0] is sentinel
