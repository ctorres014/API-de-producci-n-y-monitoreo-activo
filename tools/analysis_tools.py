"""
Analysis tools for the Analysis Agent.

Provides sentiment analysis, math evaluation, data validation,
and text summarization with graceful fallbacks when optional
dependencies (TextBlob) are not installed.
"""

import ast
import json
import math
import re
from typing import Union

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Sentiment Analyzer
# ---------------------------------------------------------------------------

@tool
def sentiment_analyzer(text: str) -> str:
    """
    Analyze the sentiment of a given text.

    Uses TextBlob when available for accurate polarity and subjectivity scores.
    Falls back to a keyword-based heuristic otherwise.

    Args:
        text: The text to analyze.

    Returns:
        A JSON string with keys:
            - polarity (float, -1.0 to 1.0): negative to positive sentiment
            - subjectivity (float, 0.0 to 1.0): objective to subjective
            - label (str): "Positive", "Negative", or "Neutral"
            - confidence (float, 0.0 to 1.0): confidence in the classification
    """
    if not text or not text.strip():
        result = {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "label": "Neutral",
            "confidence": 1.0,
            "note": "Empty text provided.",
        }
        return json.dumps(result)

    try:
        from textblob import TextBlob  # type: ignore

        blob = TextBlob(text)
        polarity = round(float(blob.sentiment.polarity), 4)
        subjectivity = round(float(blob.sentiment.subjectivity), 4)

        if polarity > 0.05:
            label = "Positive"
            confidence = round(min(1.0, 0.5 + polarity / 2), 4)
        elif polarity < -0.05:
            label = "Negative"
            confidence = round(min(1.0, 0.5 + abs(polarity) / 2), 4)
        else:
            label = "Neutral"
            confidence = round(1.0 - abs(polarity) * 5, 4)

        result = {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "label": label,
            "confidence": confidence,
        }
        return json.dumps(result)

    except ImportError:
        pass  # Fall through to keyword-based approach

    # Keyword-based fallback
    return _keyword_sentiment(text)


def _keyword_sentiment(text: str) -> str:
    """Simple keyword-based sentiment analysis fallback."""
    positive_words = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "like", "best", "happy", "joy", "awesome", "superb",
        "outstanding", "brilliant", "perfect", "positive", "beautiful",
        "nice", "pleasant", "enjoy", "thrilled", "delighted", "impressive",
    }
    negative_words = {
        "bad", "terrible", "awful", "horrible", "hate", "dislike", "worst",
        "sad", "angry", "poor", "disappointing", "disgusting", "negative",
        "ugly", "unpleasant", "frustrating", "annoying", "failed", "failure",
        "wrong", "broken", "useless", "pathetic", "dreadful",
    }

    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return json.dumps({"polarity": 0.0, "subjectivity": 0.5, "label": "Neutral", "confidence": 0.5})

    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    total = len(words)

    polarity = round((pos_count - neg_count) / max(total, 1), 4)
    subjectivity = round((pos_count + neg_count) / max(total, 1), 4)

    if polarity > 0.02:
        label = "Positive"
        confidence = round(min(1.0, 0.5 + polarity * 5), 4)
    elif polarity < -0.02:
        label = "Negative"
        confidence = round(min(1.0, 0.5 + abs(polarity) * 5), 4)
    else:
        label = "Neutral"
        confidence = round(max(0.5, 1.0 - abs(polarity) * 10), 4)

    result = {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "label": label,
        "confidence": confidence,
        "method": "keyword-based (TextBlob not installed)",
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Math Calculator
# ---------------------------------------------------------------------------

# Safe math functions exposed to the expression evaluator
_SAFE_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "pow": pow,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "tau": math.tau,
}

def _build_allowed_node_types() -> tuple:
    """Build the tuple of allowed AST node types, handling version differences."""
    types = [
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Name,
        ast.Load,
    ]
    # ast.Num was removed in Python 3.14 (deprecated since 3.8)
    if hasattr(ast, "Num"):
        types.append(ast.Num)  # type: ignore[attr-defined]
    return tuple(types)


_ALLOWED_NODE_TYPES = _build_allowed_node_types()


def _validate_ast(node: ast.AST) -> None:
    """
    Recursively validate that an AST contains only safe node types.

    Raises:
        ValueError: If the AST contains a disallowed node type.
    """
    if not isinstance(node, _ALLOWED_NODE_TYPES):
        raise ValueError(f"Disallowed operation: {type(node).__name__}")
    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


@tool
def math_calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression and return the result.

    Supports basic arithmetic (+, -, *, /, //, %, **) and common math functions
    (sqrt, abs, log, sin, cos, tan, ceil, floor, factorial, etc.).
    Dangerous operations (imports, attribute access, function calls to non-math
    functions) are blocked.

    Args:
        expression: A mathematical expression as a string, e.g. "sqrt(16) + 2**3".

    Returns:
        The numeric result as a string, or an error message if evaluation fails.
    """
    if not expression or not expression.strip():
        return "Error: Empty expression provided."

    expr = expression.strip()

    # Block obviously dangerous patterns before parsing
    danger_patterns = [
        "__",
        "import",
        "exec",
        "eval",
        "open",
        "os.",
        "sys.",
        "subprocess",
        "builtins",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "compile",
        "input",
        "print",
    ]
    expr_lower = expr.lower()
    for pattern in danger_patterns:
        if pattern in expr_lower:
            return f"Error: Expression contains disallowed pattern '{pattern}'."

    try:
        tree = ast.parse(expr, mode="eval")
        _validate_ast(tree)

        # Validate that any Name node refers to a known safe function/constant
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in _SAFE_MATH_FUNCS:
                return f"Error: Unknown name '{node.id}'. Allowed names: {sorted(_SAFE_MATH_FUNCS.keys())}"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in _SAFE_MATH_FUNCS:
                    return f"Error: Unknown function '{node.func.id}'."

        result = eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, _SAFE_MATH_FUNCS)  # noqa: S307

        # Format the result nicely
        if isinstance(result, float) and result.is_integer():
            return str(result)
        return str(result)

    except (ValueError, TypeError, ZeroDivisionError, OverflowError, SyntaxError) as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# ---------------------------------------------------------------------------
# Data Validator
# ---------------------------------------------------------------------------

@tool
def data_validator(data: str, schema_description: str) -> str:
    """
    Validate JSON data against a natural-language schema description.

    Parses the JSON from `data`, then checks for required fields mentioned
    in `schema_description` and validates basic type constraints when
    expressed in the description.

    Args:
        data: A JSON string to validate.
        schema_description: A natural-language description of expected fields
            and types, e.g. "requires name (string), age (number), email".

    Returns:
        A JSON string with keys:
            - valid (bool): Whether validation passed.
            - errors (list[str]): List of validation errors found.
            - warnings (list[str]): Non-fatal observations.
            - summary (str): Human-readable validation summary.
    """
    errors: list = []
    warnings: list = []

    # --- Parse JSON ---
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        result = {
            "valid": False,
            "errors": [f"Invalid JSON: {exc}"],
            "warnings": [],
            "summary": "Validation failed: the provided data is not valid JSON.",
        }
        return json.dumps(result)

    if not isinstance(parsed, dict):
        result = {
            "valid": False,
            "errors": ["Data must be a JSON object (dict), not a list or primitive."],
            "warnings": [],
            "summary": "Validation failed: expected a JSON object.",
        }
        return json.dumps(result)

    # --- Extract field requirements from schema description ---
    # Look for patterns like "field_name (type)" or "field_name as type" or just "field_name"
    field_pattern = re.findall(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(([^)]*)\)|\bas\b\s*(\w+))?",
        schema_description,
    )

    required_fields: dict = {}
    skip_words = {
        "requires", "required", "should", "must", "have", "has", "contain",
        "contains", "include", "includes", "and", "or", "the", "with", "a",
        "an", "is", "are", "be", "that", "schema", "field", "fields",
        "object", "data", "json",
    }

    for name, paren_type, as_type in field_pattern:
        if name.lower() in skip_words:
            continue
        field_type = (paren_type or as_type or "").strip().lower()
        required_fields[name] = field_type

    # --- Check required fields ---
    for field_name, expected_type in required_fields.items():
        if field_name not in parsed:
            errors.append(f"Missing required field: '{field_name}'")
            continue

        value = parsed[field_name]

        # Type checks
        if expected_type in ("string", "str"):
            if not isinstance(value, str):
                errors.append(f"Field '{field_name}' should be a string, got {type(value).__name__}.")
        elif expected_type in ("number", "num", "int", "integer", "float"):
            if not isinstance(value, (int, float)):
                errors.append(f"Field '{field_name}' should be a number, got {type(value).__name__}.")
        elif expected_type in ("bool", "boolean"):
            if not isinstance(value, bool):
                errors.append(f"Field '{field_name}' should be a boolean, got {type(value).__name__}.")
        elif expected_type in ("list", "array"):
            if not isinstance(value, list):
                errors.append(f"Field '{field_name}' should be a list/array, got {type(value).__name__}.")
        elif expected_type in ("object", "dict"):
            if not isinstance(value, dict):
                errors.append(f"Field '{field_name}' should be an object/dict, got {type(value).__name__}.")
        elif expected_type == "":
            # No type specified — just check presence (already done above)
            pass

    # --- Additional warnings for null/empty values ---
    for key, val in parsed.items():
        if val is None:
            warnings.append(f"Field '{key}' is null.")
        elif val == "" or val == [] or val == {}:
            warnings.append(f"Field '{key}' is empty.")

    valid = len(errors) == 0
    found_fields = list(parsed.keys())
    summary = (
        f"Validation {'passed' if valid else 'failed'}. "
        f"Found fields: {found_fields}. "
        f"Errors: {len(errors)}, Warnings: {len(warnings)}."
    )

    result = {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Text Summarizer
# ---------------------------------------------------------------------------

@tool
def text_summarizer(text: str, max_words: int = 100) -> str:
    """
    Summarize a piece of text to approximately max_words words.

    Uses an extractive approach: selects the most informative sentences
    based on word frequency scoring, then truncates to max_words.

    Args:
        text: The text to summarize.
        max_words: Maximum number of words in the summary (default: 100).

    Returns:
        A summarized version of the text.
    """
    if not text or not text.strip():
        return "No text provided to summarize."

    max_words = max(10, int(max_words))

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text[:max_words * 6]  # Rough character estimate

    # If already short enough, return as-is
    word_count = len(text.split())
    if word_count <= max_words:
        return text

    # Score sentences by word frequency (simple extractive approach)
    # Build word frequency map (excluding stop words)
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "this", "that", "these",
        "those", "it", "its", "as", "if", "so", "not", "no", "nor", "yet",
    }

    words = re.findall(r"\b[a-z]+\b", text.lower())
    freq: dict = {}
    for w in words:
        if w not in stop_words:
            freq[w] = freq.get(w, 0) + 1

    # Score each sentence
    def score_sentence(sent: str) -> float:
        sent_words = re.findall(r"\b[a-z]+\b", sent.lower())
        if not sent_words:
            return 0.0
        return sum(freq.get(w, 0) for w in sent_words) / len(sent_words)

    scored = sorted(enumerate(sentences), key=lambda t: score_sentence(t[1]), reverse=True)

    # Select top sentences in original order until we hit max_words
    selected_indices = set()
    current_words = 0
    for idx, sent in scored:
        sent_words = len(sent.split())
        if current_words + sent_words <= max_words:
            selected_indices.add(idx)
            current_words += sent_words
        if current_words >= max_words:
            break

    # Fallback: take first N sentences if scoring selected nothing
    if not selected_indices:
        selected_indices = {0}

    # Reassemble in original order
    summary_sentences = [sentences[i] for i in sorted(selected_indices)]
    summary = " ".join(summary_sentences)

    # Hard truncate if still over limit
    final_words = summary.split()
    if len(final_words) > max_words:
        summary = " ".join(final_words[:max_words]) + "..."

    return summary
