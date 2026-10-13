"""
Shared fixtures for the Entrega 7 test suite.
"""

import json
import pytest
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# FakeRedis — in-memory stand-in for redis.Redis
# ---------------------------------------------------------------------------

class FakeRedis:
    """
    Minimal Redis emulation backed by an in-memory dict.

    Implements only the methods used by app/worker.py so tests don't need a
    running Redis instance.
    """

    def __init__(self):
        self._store: dict = {}

    def setex(self, key, ttl, value):
        self._store[key] = value

    def get(self, key):
        return self._store.get(key)

    def ping(self):
        return True

    def close(self):
        pass

    # Helper used in tests to inspect raw stored value
    def get_parsed(self, key) -> dict | None:
        raw = self._store.get(key)
        return json.loads(raw) if raw is not None else None


@pytest.fixture
def fake_redis():
    """Return a fresh FakeRedis instance per test."""
    return FakeRedis()


# ---------------------------------------------------------------------------
# Base OrchestratorState builder
# ---------------------------------------------------------------------------

def make_state(**overrides) -> dict:
    """
    Return a fully populated OrchestratorState dict.

    All fields (including the Entrega 7 additions) are present so tests
    never fail due to missing keys.
    """
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
        "is_critical": False,
        "is_approved": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def base_state():
    """Return a minimal valid OrchestratorState dict."""
    return make_state()
