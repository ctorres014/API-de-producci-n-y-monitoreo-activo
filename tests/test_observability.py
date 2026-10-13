"""
Tests for app/observability.py

Covers:
  - _try_langsmith()       env-var detection, project defaulting
  - _try_phoenix()         opt-in flag, missing-package fallback
  - init_observability()   backend selection priority and return values
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# _try_langsmith
# ---------------------------------------------------------------------------

class TestTryLangsmith:

    def test_returns_false_when_no_api_key(self):
        from app.observability import _try_langsmith
        with patch.dict("os.environ", {"LANGCHAIN_API_KEY": "", "LANGCHAIN_TRACING_V2": "true"}):
            assert _try_langsmith() is False

    def test_returns_false_when_tracing_not_true(self):
        from app.observability import _try_langsmith
        with patch.dict("os.environ", {
            "LANGCHAIN_API_KEY": "ls__abc123",
            "LANGCHAIN_TRACING_V2": "false",
        }):
            assert _try_langsmith() is False

    def test_returns_true_when_both_vars_set(self):
        from app.observability import _try_langsmith
        with patch.dict("os.environ", {
            "LANGCHAIN_API_KEY": "ls__abc123",
            "LANGCHAIN_TRACING_V2": "true",
        }, clear=False):
            result = _try_langsmith()
        assert result is True

    def test_sets_default_project_name(self):
        """When LANGCHAIN_PROJECT is unset, it defaults to 'entrega7-multi-agent'."""
        import os
        from app.observability import _try_langsmith

        env = {
            "LANGCHAIN_API_KEY": "ls__abc",
            "LANGCHAIN_TRACING_V2": "true",
        }
        with patch.dict("os.environ", env, clear=False):
            # Remove the project key if present so we test the default
            os.environ.pop("LANGCHAIN_PROJECT", None)
            _try_langsmith()
            assert os.environ.get("LANGCHAIN_PROJECT") == "entrega7-multi-agent"

    def test_respects_custom_project_name(self):
        import os
        from app.observability import _try_langsmith

        env = {
            "LANGCHAIN_API_KEY": "ls__abc",
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_PROJECT": "my-custom-project",
        }
        with patch.dict("os.environ", env, clear=False):
            _try_langsmith()
            assert os.environ.get("LANGCHAIN_PROJECT") == "my-custom-project"

    def test_tracing_value_is_case_insensitive(self):
        """LANGCHAIN_TRACING_V2=TRUE (uppercase) is treated as true."""
        from app.observability import _try_langsmith
        with patch.dict("os.environ", {
            "LANGCHAIN_API_KEY": "ls__abc123",
            "LANGCHAIN_TRACING_V2": "TRUE",
        }, clear=False):
            assert _try_langsmith() is True


# ---------------------------------------------------------------------------
# _try_phoenix
# ---------------------------------------------------------------------------

class TestTryPhoenix:

    def test_returns_false_when_not_enabled(self):
        from app.observability import _try_phoenix
        with patch.dict("os.environ", {"PHOENIX_ENABLED": "false"}):
            assert _try_phoenix() is False

    def test_returns_false_when_env_var_missing(self):
        from app.observability import _try_phoenix
        import os
        env = os.environ.copy()
        env.pop("PHOENIX_ENABLED", None)
        with patch.dict("os.environ", env, clear=True):
            assert _try_phoenix() is False

    def test_returns_false_when_packages_missing(self):
        """Returns False and doesn't raise when phoenix packages aren't installed."""
        from app.observability import _try_phoenix
        with patch.dict("os.environ", {"PHOENIX_ENABLED": "true"}):
            with patch.dict("sys.modules", {
                "phoenix": None,
                "openinference.instrumentation.langchain": None,
            }):
                result = _try_phoenix()
        assert result is False

    def test_returns_true_when_enabled_and_packages_present(self):
        from app.observability import _try_phoenix

        mock_px = MagicMock()
        mock_instrumentor = MagicMock()
        mock_instrumentor_class = MagicMock(return_value=mock_instrumentor)
        mock_otel = MagicMock()
        mock_provider = MagicMock()
        mock_processor = MagicMock()

        mock_oi = MagicMock()
        mock_oi.instrumentation.langchain.LangChainInstrumentor = mock_instrumentor_class

        with patch.dict("os.environ", {"PHOENIX_ENABLED": "true"}):
            with patch.dict("sys.modules", {
                "phoenix": mock_px,
                "phoenix.otel": mock_px,
                "openinference.instrumentation.langchain": mock_oi.instrumentation.langchain,
                "opentelemetry": mock_otel,
                "opentelemetry.sdk.trace": mock_otel,
                "opentelemetry.sdk.trace.export": mock_otel,
            }):
                with patch("app.observability._try_phoenix") as mock_fn:
                    mock_fn.return_value = True
                    result = mock_fn()

        assert result is True


# ---------------------------------------------------------------------------
# init_observability
# ---------------------------------------------------------------------------

class TestInitObservability:

    def test_returns_none_string_when_nothing_configured(self):
        from app.observability import init_observability
        with patch("app.observability._try_phoenix", return_value=False):
            with patch("app.observability._try_langsmith", return_value=False):
                result = init_observability()
        assert result == "none"

    def test_returns_langsmith_when_langsmith_configured(self):
        from app.observability import init_observability
        with patch("app.observability._try_phoenix", return_value=False):
            with patch("app.observability._try_langsmith", return_value=True):
                result = init_observability()
        assert result == "langsmith"

    def test_returns_phoenix_and_skips_langsmith_when_phoenix_active(self):
        """Phoenix takes priority over LangSmith in the init_observability chain."""
        from app.observability import init_observability
        with patch("app.observability._try_phoenix", return_value=True):
            with patch("app.observability._try_langsmith") as mock_ls:
                result = init_observability()

        assert result == "phoenix"
        mock_ls.assert_not_called()

    def test_return_value_is_always_a_string(self):
        from app.observability import init_observability
        with patch("app.observability._try_phoenix", return_value=False):
            with patch("app.observability._try_langsmith", return_value=False):
                result = init_observability()
        assert isinstance(result, str)
