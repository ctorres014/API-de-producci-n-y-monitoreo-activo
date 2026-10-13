"""
Configuration module for the multi-agent orchestrator.

Reads LLM provider and model settings from environment variables,
supporting Anthropic, OpenAI, and Ollama backends.
"""

import os
from typing import Any


# Maximum number of supervisor iterations before forcing FINISH
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "6"))


def get_llm() -> Any:
    """
    Instantiate and return an LLM based on the LLM_PROVIDER environment variable.

    Supported providers:
        - "anthropic" (default): Uses ChatAnthropic with ANTHROPIC_MODEL env var.
        - "openai": Uses ChatOpenAI with OPENAI_MODEL env var.
        - "ollama": Uses ChatOllama with OLLAMA_MODEL and OLLAMA_BASE_URL env vars.

    Returns:
        A LangChain chat model instance.

    Raises:
        ValueError: If an unsupported LLM_PROVIDER is specified.
        ImportError: If the required package for the chosen provider is not installed.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is not installed. "
                "Run: pip install langchain-anthropic"
            ) from exc

        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        kwargs: dict = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is not installed. "
                "Run: pip install langchain-openai"
            ) from exc

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "langchain-ollama is not installed. "
                "Run: pip install langchain-ollama"
            ) from exc

        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Choose one of: anthropic, openai, ollama"
        )
