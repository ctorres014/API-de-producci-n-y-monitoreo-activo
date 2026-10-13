"""
Tests for all tools in the tools package.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Sentiment Analyzer Tests
# ---------------------------------------------------------------------------

class TestSentimentAnalyzer:
    """Tests for the sentiment_analyzer tool."""

    def test_sentiment_analyzer_positive(self):
        """Positive text is classified as Positive."""
        from tools.analysis_tools import sentiment_analyzer

        result_str = sentiment_analyzer.invoke("I love this amazing product, it is wonderful!")
        result = json.loads(result_str)

        assert "polarity" in result
        assert "subjectivity" in result
        assert "label" in result
        assert "confidence" in result
        assert result["label"] == "Positive", f"Expected Positive, got {result['label']}"
        assert result["polarity"] > 0

    def test_sentiment_analyzer_negative(self):
        """Negative text is classified as Negative."""
        from tools.analysis_tools import sentiment_analyzer

        result_str = sentiment_analyzer.invoke("This is terrible and awful, I hate it!")
        result = json.loads(result_str)

        assert result["label"] == "Negative", f"Expected Negative, got {result['label']}"
        assert result["polarity"] < 0

    def test_sentiment_analyzer_neutral(self):
        """Neutral text produces a score near zero."""
        from tools.analysis_tools import sentiment_analyzer

        result_str = sentiment_analyzer.invoke("The document is on the table.")
        result = json.loads(result_str)

        assert "label" in result
        assert result["label"] in ("Neutral", "Positive", "Negative")
        # Polarity should be close to zero for neutral text
        assert -0.5 <= result["polarity"] <= 0.5

    def test_sentiment_analyzer_returns_valid_json(self):
        """sentiment_analyzer always returns valid JSON."""
        from tools.analysis_tools import sentiment_analyzer

        result_str = sentiment_analyzer.invoke("Some random text for testing purposes.")
        result = json.loads(result_str)  # Should not raise

        assert isinstance(result, dict)
        assert isinstance(result.get("polarity"), float)
        assert isinstance(result.get("confidence"), float)

    def test_sentiment_analyzer_empty_text(self):
        """Empty text returns Neutral with confidence 1.0."""
        from tools.analysis_tools import sentiment_analyzer

        result_str = sentiment_analyzer.invoke("")
        result = json.loads(result_str)

        assert result["label"] == "Neutral"

    def test_sentiment_analyzer_polarity_range(self):
        """Polarity is always in [-1, 1] range."""
        from tools.analysis_tools import sentiment_analyzer

        texts = [
            "Absolutely fantastic and brilliant!",
            "Dreadful and pathetic failure.",
            "The sky is blue.",
        ]
        for text in texts:
            result = json.loads(sentiment_analyzer.invoke(text))
            assert -1.0 <= result["polarity"] <= 1.0, f"Polarity out of range for: {text}"

    def test_sentiment_analyzer_subjectivity_range(self):
        """Subjectivity is always in [0, 1] range."""
        from tools.analysis_tools import sentiment_analyzer

        result = json.loads(sentiment_analyzer.invoke("I think this might be good."))
        assert 0.0 <= result["subjectivity"] <= 1.0


# ---------------------------------------------------------------------------
# Math Calculator Tests
# ---------------------------------------------------------------------------

class TestMathCalculator:
    """Tests for the math_calculator tool."""

    def test_math_calculator_basic_addition(self):
        """'2 + 2' evaluates to '4'."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("2 + 2")
        assert result == "4", f"Expected '4', got '{result}'"

    def test_math_calculator_basic_subtraction(self):
        """'10 - 3' evaluates to '7'."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("10 - 3")
        assert result == "7"

    def test_math_calculator_multiplication(self):
        """'3 * 4' evaluates to '12'."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("3 * 4")
        assert result == "12"

    def test_math_calculator_division(self):
        """'10 / 4' evaluates to '2.5'."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("10 / 4")
        assert result == "2.5"

    def test_math_calculator_complex_sqrt(self):
        """'sqrt(16)' evaluates to '4.0'."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("sqrt(16)")
        assert result == "4.0", f"Expected '4.0', got '{result}'"

    def test_math_calculator_power(self):
        """'2 ** 8' evaluates to '256'."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("2 ** 8")
        assert result == "256"

    def test_math_calculator_complex_expression(self):
        """Complex expression with math functions evaluates correctly."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("sqrt(16) + 2**3")
        # sqrt(16) = 4.0, 2**3 = 8, total = 12.0
        assert result == "12.0", f"Expected '12.0', got '{result}'"

    def test_math_calculator_modulo(self):
        """'17 % 5' evaluates to '2'."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("17 % 5")
        assert result == "2"

    def test_math_calculator_invalid_import(self):
        """Dangerous __import__ call returns error."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("__import__('os')")
        assert "Error" in result, f"Expected error for dangerous input, got: {result}"

    def test_math_calculator_invalid_exec(self):
        """'exec' in expression returns error."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("exec('import os')")
        assert "Error" in result

    def test_math_calculator_division_by_zero(self):
        """Division by zero returns an error message."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("1 / 0")
        assert "Error" in result

    def test_math_calculator_unknown_function(self):
        """Unknown function names return an error."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("unknown_func(5)")
        assert "Error" in result

    def test_math_calculator_pi_constant(self):
        """'pi' constant is accessible."""
        from tools.analysis_tools import math_calculator
        import math

        result = math_calculator.invoke("round(pi, 5)")
        assert result == str(round(math.pi, 5))

    def test_math_calculator_empty_expression(self):
        """Empty expression returns error."""
        from tools.analysis_tools import math_calculator

        result = math_calculator.invoke("")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Data Validator Tests
# ---------------------------------------------------------------------------

class TestDataValidator:
    """Tests for the data_validator tool."""

    def test_data_validator_valid_json(self):
        """Valid JSON matching the schema passes validation."""
        from tools.analysis_tools import data_validator

        data = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
        schema = "requires name (string), age (number), email (string)"

        result_str = data_validator.invoke({"data": data, "schema_description": schema})
        result = json.loads(result_str)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_data_validator_missing_field(self):
        """JSON missing a required field fails validation."""
        from tools.analysis_tools import data_validator

        data = '{"name": "Alice"}'
        schema = "requires name (string), age (number)"

        result_str = data_validator.invoke({"data": data, "schema_description": schema})
        result = json.loads(result_str)

        assert result["valid"] is False
        assert any("age" in err for err in result["errors"])

    def test_data_validator_invalid_json(self):
        """Invalid JSON string fails with parsing error."""
        from tools.analysis_tools import data_validator

        data = "this is not json {"
        schema = "requires name"

        result_str = data_validator.invoke({"data": data, "schema_description": schema})
        result = json.loads(result_str)

        assert result["valid"] is False
        assert any("JSON" in err or "json" in err.lower() for err in result["errors"])

    def test_data_validator_wrong_type(self):
        """Field with wrong type produces a type error."""
        from tools.analysis_tools import data_validator

        data = '{"name": 123, "age": 30}'
        schema = "requires name (string), age (number)"

        result_str = data_validator.invoke({"data": data, "schema_description": schema})
        result = json.loads(result_str)

        assert result["valid"] is False
        assert any("name" in err for err in result["errors"])

    def test_data_validator_empty_json_object(self):
        """Empty JSON object fails when fields are required."""
        from tools.analysis_tools import data_validator

        data = "{}"
        schema = "requires company (string), revenue (number)"

        result_str = data_validator.invoke({"data": data, "schema_description": schema})
        result = json.loads(result_str)

        assert result["valid"] is False
        assert len(result["errors"]) >= 2

    def test_data_validator_returns_summary(self):
        """Validation result always includes a summary string."""
        from tools.analysis_tools import data_validator

        data = '{"x": 1}'
        schema = "requires x"

        result_str = data_validator.invoke({"data": data, "schema_description": schema})
        result = json.loads(result_str)

        assert "summary" in result
        assert isinstance(result["summary"], str)

    def test_data_validator_null_values_produce_warnings(self):
        """Null field values produce warnings."""
        from tools.analysis_tools import data_validator

        data = '{"name": "Alice", "age": null}'
        schema = "requires name (string)"

        result_str = data_validator.invoke({"data": data, "schema_description": schema})
        result = json.loads(result_str)

        assert len(result["warnings"]) > 0


# ---------------------------------------------------------------------------
# Web Search Tests
# ---------------------------------------------------------------------------

class TestWebSearch:
    """Tests for the web_search tool."""

    def test_web_search_returns_string(self):
        """web_search always returns a string."""
        from tools.search_tools import web_search

        # Without TAVILY_API_KEY set, should return simulated results
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
            result = web_search.invoke("Python programming language")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_web_search_simulated_when_no_key(self):
        """web_search returns simulated note when TAVILY_API_KEY is not set."""
        from tools.search_tools import web_search

        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
            result = web_search.invoke("artificial intelligence")

        assert "SIMULATED" in result or "simulated" in result.lower() or "artificial intelligence" in result.lower()

    def test_web_search_contains_query(self):
        """Simulated web search output references the original query."""
        from tools.search_tools import web_search

        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
            result = web_search.invoke("quantum computing")

        assert "quantum computing" in result.lower() or "Simulated" in result

    def test_web_search_with_mocked_tavily(self):
        """web_search uses TavilySearch when API key is present (mocked)."""
        from tools.search_tools import web_search

        mock_results = [
            {"title": "Test Article", "url": "https://example.com", "content": "Test content."}
        ]

        mock_tavily_module = MagicMock()
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_results
        mock_tavily_module.TavilySearch.return_value = mock_instance

        with patch.dict("os.environ", {"TAVILY_API_KEY": "fake-key-for-testing"}):
            with patch.dict("sys.modules", {"langchain_tavily": mock_tavily_module}):
                result = web_search.invoke("test query")

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Wikipedia Search Tests
# ---------------------------------------------------------------------------

class TestWikipediaSearch:
    """Tests for the wikipedia_search tool."""

    def test_wikipedia_search_returns_string(self):
        """wikipedia_search always returns a string."""
        from tools.search_tools import wikipedia_search

        result = wikipedia_search.invoke("Python programming language")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Text Summarizer Tests
# ---------------------------------------------------------------------------

class TestTextSummarizer:
    """Tests for the text_summarizer tool."""

    def test_text_summarizer_basic(self):
        """text_summarizer returns a non-empty string for normal text."""
        from tools.analysis_tools import text_summarizer

        long_text = (
            "Artificial intelligence is a broad field of computer science. "
            "It focuses on creating systems that can perform tasks that normally require human intelligence. "
            "These tasks include visual perception, speech recognition, decision-making, and language translation. "
            "Machine learning is a subset of AI that enables systems to learn from data. "
            "Deep learning uses neural networks to model complex patterns. "
            "Natural language processing allows machines to understand and generate human language. "
            "Computer vision enables machines to interpret visual information from the world. "
            "Reinforcement learning trains agents to make decisions by rewarding desired behaviors."
        )

        result = text_summarizer.invoke({"text": long_text, "max_words": 50})
        assert isinstance(result, str)
        assert len(result) > 0
        word_count = len(result.split())
        assert word_count <= 60, f"Summary too long: {word_count} words"

    def test_text_summarizer_short_text_unchanged(self):
        """Short text that is already under max_words is returned as-is."""
        from tools.analysis_tools import text_summarizer

        short_text = "AI is amazing."
        result = text_summarizer.invoke({"text": short_text, "max_words": 100})
        assert result == short_text

    def test_text_summarizer_respects_max_words(self):
        """Summary stays within max_words limit (with small tolerance for ellipsis)."""
        from tools.analysis_tools import text_summarizer

        long_text = " ".join(["word"] * 500)
        result = text_summarizer.invoke({"text": long_text, "max_words": 30})
        # Allow a few extra words for sentence boundary rounding
        word_count = len(result.replace("...", "").split())
        assert word_count <= 35

    def test_text_summarizer_empty_text(self):
        """Empty text returns a descriptive message."""
        from tools.analysis_tools import text_summarizer

        result = text_summarizer.invoke({"text": "", "max_words": 50})
        assert "No text" in result or result == ""

    def test_text_summarizer_default_max_words(self):
        """text_summarizer works with default max_words (100)."""
        from tools.analysis_tools import text_summarizer

        text = "This is a test. " * 50
        result = text_summarizer.invoke({"text": text})
        assert isinstance(result, str)
        assert len(result) > 0
