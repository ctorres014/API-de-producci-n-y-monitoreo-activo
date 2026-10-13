"""
Tests for app/main.py — FastAPI endpoints

Covers:
  GET  /health
  POST /tasks
  GET  /tasks/{job_id}
  POST /tasks/{job_id}/approve

All tests use FastAPI's TestClient with mocked Redis and graph so no
external services are required.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from fastapi.testclient import TestClient

from tests.conftest import FakeRedis, make_state
from app.worker import (
    create_job,
    update_job,
    STATUS_AWAITING,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_RUNNING,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def mock_graph():
    g = MagicMock()
    g.invoke.return_value = make_state(task_completed=True)
    return g


@pytest.fixture
def api_client(fake_redis, mock_graph):
    """
    TestClient with all external dependencies mocked.

    Patches:
      - get_redis_checkpointer → MemorySaver (no Redis needed for graph)
      - build_api_graph         → mock_graph
      - redis_lib.from_url      → fake_redis
      - init_observability      → "none"
      - run_in_threadpool       → no-op so background tasks don't run the graph
    """
    from langgraph.checkpoint.memory import MemorySaver
    import app.main as app_main

    # Prevent background tasks from actually executing the graph
    async def noop_threadpool(func, *args, **kwargs):
        pass

    with (
        patch("app.main.get_redis_checkpointer", return_value=MemorySaver()),
        patch("app.main.build_api_graph", return_value=mock_graph),
        patch("app.main.redis_lib.from_url", return_value=fake_redis),
        patch("app.main.init_observability", return_value="none"),
        patch("app.main.run_in_threadpool", new=noop_threadpool),
    ):
        with TestClient(app_main.app, raise_server_exceptions=True) as client:
            yield client


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_returns_200(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, api_client):
        data = api_client.get("/health").json()
        assert data["status"] == "ok"

    def test_reports_redis_connected(self, api_client):
        data = api_client.get("/health").json()
        assert data["redis"] == "connected"

    def test_reports_graph_compiled(self, api_client):
        data = api_client.get("/health").json()
        assert data["graph"] == "compiled"


# ---------------------------------------------------------------------------
# POST /tasks
# ---------------------------------------------------------------------------

class TestSubmitTask:

    def test_returns_202(self, api_client):
        response = api_client.post("/tasks", json={"query": "What is AI?"})
        assert response.status_code == 202

    def test_returns_job_id(self, api_client):
        data = api_client.post("/tasks", json={"query": "What is AI?"}).json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0

    def test_returns_pending_status(self, api_client):
        data = api_client.post("/tasks", json={"query": "What is AI?"}).json()
        assert data["status"] == "pending"

    def test_job_is_retrievable_after_submission(self, api_client):
        post_data = api_client.post("/tasks", json={"query": "Explain LLMs"}).json()
        job_id = post_data["job_id"]

        get_data = api_client.get(f"/tasks/{job_id}").json()
        assert get_data["job_id"] == job_id

    def test_rejects_empty_query(self, api_client):
        """query must be at least 3 characters (Pydantic min_length=3)."""
        response = api_client.post("/tasks", json={"query": "AB"})
        assert response.status_code == 422

    def test_rejects_missing_query_field(self, api_client):
        response = api_client.post("/tasks", json={})
        assert response.status_code == 422

    def test_different_queries_get_different_job_ids(self, api_client):
        id1 = api_client.post("/tasks", json={"query": "Query one here"}).json()["job_id"]
        id2 = api_client.post("/tasks", json={"query": "Query two here"}).json()["job_id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# GET /tasks/{job_id}
# ---------------------------------------------------------------------------

class TestGetTaskStatus:

    def test_returns_404_for_unknown_job(self, api_client):
        response = api_client.get("/tasks/nonexistent-job-id")
        assert response.status_code == 404

    def test_returns_404_detail_message(self, api_client):
        data = api_client.get("/tasks/ghost-id").json()
        assert "not found" in data["detail"].lower()

    def test_returns_200_for_existing_job(self, api_client):
        job_id = api_client.post("/tasks", json={"query": "Test query one"}).json()["job_id"]
        response = api_client.get(f"/tasks/{job_id}")
        assert response.status_code == 200

    def test_response_contains_required_fields(self, api_client):
        job_id = api_client.post("/tasks", json={"query": "Test query two"}).json()["job_id"]
        data = api_client.get(f"/tasks/{job_id}").json()

        for field in ("job_id", "status", "query", "created_at", "updated_at"):
            assert field in data, f"Missing field: {field}"

    def test_query_is_preserved_in_response(self, api_client):
        query = "What is deep learning?"
        job_id = api_client.post("/tasks", json={"query": query}).json()["job_id"]
        data = api_client.get(f"/tasks/{job_id}").json()
        assert data["query"] == query


# ---------------------------------------------------------------------------
# POST /tasks/{job_id}/approve
# ---------------------------------------------------------------------------

class TestApproveTask:

    def _create_awaiting_job(self, api_client, fake_redis) -> str:
        """Submit a job and manually set its status to AWAITING_APPROVAL."""
        job_id = api_client.post("/tasks", json={"query": "Delete the records"}).json()["job_id"]
        update_job(fake_redis, job_id, status=STATUS_AWAITING)
        return job_id

    def test_returns_404_for_unknown_job(self, api_client):
        response = api_client.post(
            "/tasks/ghost-id/approve",
            json={"approved": True},
        )
        assert response.status_code == 404

    def test_returns_409_when_job_not_awaiting(self, api_client, fake_redis):
        """Approving a job that isn't AWAITING_APPROVAL returns 409 Conflict."""
        job_id = api_client.post("/tasks", json={"query": "Normal query here"}).json()["job_id"]
        # Status is still PENDING — not awaiting
        response = api_client.post(f"/tasks/{job_id}/approve", json={"approved": True})
        assert response.status_code == 409

    def test_409_detail_mentions_expected_status(self, api_client):
        job_id = api_client.post("/tasks", json={"query": "Normal query again"}).json()["job_id"]
        data = api_client.post(f"/tasks/{job_id}/approve", json={"approved": True}).json()
        assert "awaiting_approval" in data["detail"].lower()

    def test_returns_200_when_job_is_awaiting(self, api_client, fake_redis):
        job_id = self._create_awaiting_job(api_client, fake_redis)
        response = api_client.post(f"/tasks/{job_id}/approve", json={"approved": True})
        assert response.status_code == 200

    def test_approve_true_returns_running_status(self, api_client, fake_redis):
        job_id = self._create_awaiting_job(api_client, fake_redis)
        data = api_client.post(f"/tasks/{job_id}/approve", json={"approved": True}).json()
        assert data["status"] == "running"

    def test_approve_false_returns_running_status(self, api_client, fake_redis):
        job_id = self._create_awaiting_job(api_client, fake_redis)
        data = api_client.post(f"/tasks/{job_id}/approve", json={"approved": False}).json()
        assert data["status"] == "running"

    def test_rejects_missing_approved_field(self, api_client, fake_redis):
        job_id = self._create_awaiting_job(api_client, fake_redis)
        response = api_client.post(f"/tasks/{job_id}/approve", json={})
        assert response.status_code == 422

    def test_job_id_echoed_in_approve_response(self, api_client, fake_redis):
        job_id = self._create_awaiting_job(api_client, fake_redis)
        data = api_client.post(f"/tasks/{job_id}/approve", json={"approved": True}).json()
        assert data["job_id"] == job_id
