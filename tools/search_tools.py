"""
Search tools for the Research Agent.

Provides web search (via Tavily when available) and Wikipedia search
with graceful fallbacks when external APIs or packages are unavailable.
"""

import os
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information about a given query.

    Uses the Tavily Search API when TAVILY_API_KEY is set in the environment.
    Falls back to a simulated response when the API key is absent or the
    langchain-tavily package is not installed.

    Args:
        query: The search query string.

    Returns:
        A string containing search results or a simulated response.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()

    if api_key:
        try:
            from langchain_tavily import TavilySearch

            search = TavilySearch(max_results=5, tavily_api_key=api_key)
            results = search.invoke(query)
            if isinstance(results, list):
                formatted = []
                for i, r in enumerate(results, 1):
                    if isinstance(r, dict):
                        title = r.get("title", "No title")
                        url = r.get("url", "")
                        content = r.get("content", "")
                        formatted.append(f"[{i}] {title}\nURL: {url}\n{content}")
                    else:
                        formatted.append(f"[{i}] {r}")
                return "\n\n".join(formatted) if formatted else "No results found."
            return str(results)

        except ImportError:
            pass  # Fall through to simulation
        except Exception as exc:
            return f"Web search encountered an error: {exc}. Returning simulated results.\n" + _simulated_web_search(query)

    # Simulate results when no API key is configured
    note = (
        "[SIMULATED] TAVILY_API_KEY is not set. "
        "These results are simulated for demonstration purposes.\n\n"
    )
    return note + _simulated_web_search(query)


def _simulated_web_search(query: str) -> str:
    """Return a simulated web search response for the given query."""
    return (
        f"Simulated web search results for: '{query}'\n\n"
        f"[1] Overview of {query}\n"
        f"    This is a comprehensive overview of the topic '{query}', "
        f"covering key aspects, recent developments, and important context.\n\n"
        f"[2] Latest developments in {query}\n"
        f"    Recent news and updates related to '{query}' show significant "
        f"activity and ongoing research in this area.\n\n"
        f"[3] Expert analysis: {query}\n"
        f"    Experts in the field have noted that '{query}' is an important "
        f"topic with broad implications across multiple domains.\n\n"
        f"Note: For real results, set the TAVILY_API_KEY environment variable."
    )


@tool
def wikipedia_search(topic: str) -> str:
    """
    Search Wikipedia for factual information about a topic.

    Uses WikipediaQueryRun from langchain_community when available.
    Falls back to a simulated response if the package is not installed.

    Args:
        topic: The topic or keyword to look up on Wikipedia.

    Returns:
        A string with the Wikipedia article summary or a simulated response.
    """
    try:
        from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
        from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

        api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)
        wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
        result = wiki_tool.run(topic)
        if result:
            return result
        return f"No Wikipedia article found for: {topic}"

    except ImportError:
        note = (
            "[SIMULATED] langchain-community or wikipedia package is not installed. "
            "These results are simulated for demonstration purposes.\n\n"
        )
        return note + _simulated_wikipedia_search(topic)
    except Exception as exc:
        return (
            f"Wikipedia search encountered an error: {exc}. "
            f"Returning simulated results.\n\n" + _simulated_wikipedia_search(topic)
        )


def _simulated_wikipedia_search(topic: str) -> str:
    """Return a simulated Wikipedia summary for the given topic."""
    return (
        f"Simulated Wikipedia summary for: '{topic}'\n\n"
        f"{topic} is a significant subject that has been studied and documented "
        f"extensively. According to available information, it encompasses various "
        f"aspects including historical context, current state, and future outlook. "
        f"The topic has relevance in multiple fields and has been the subject of "
        f"academic research, policy discussions, and public interest.\n\n"
        f"Key facts:\n"
        f"- {topic} has been documented since early in its development\n"
        f"- Multiple organizations and researchers have studied {topic}\n"
        f"- The topic continues to evolve with new developments regularly\n\n"
        f"Note: For real Wikipedia results, install langchain-community and wikipedia packages."
    )
