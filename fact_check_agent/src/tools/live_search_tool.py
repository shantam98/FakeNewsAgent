"""Live search tool — queries Tavily for current web evidence about a claim.

This is a tool, not an agent: it makes a single Tavily API call (with one
automatic retry for source diversity) and returns structured results.
No LLM call, no planning.

Tuning surface:
  - _MIN_DISTINCT_DOMAINS: minimum number of distinct source domains required
"""
import logging

from tavily import TavilyClient

logger = logging.getLogger(__name__)

_MIN_DISTINCT_DOMAINS = 3


def search_live(claim_text: str, api_key: str, max_results: int = 5) -> list[dict]:
    """Search Tavily for evidence about the claim.

    Enforces a minimum of _MIN_DISTINCT_DOMAINS distinct source domains.
    Retries with a broader query if the first pass returns fewer.
    """
    client = TavilyClient(api_key=api_key)

    results = _run_search(client, claim_text, max_results)

    distinct_domains = _count_distinct_domains(results)
    if distinct_domains < _MIN_DISTINCT_DOMAINS:
        broader_query = f"fact check: {claim_text}"
        logger.debug(
            "Only %d distinct domains — retrying with broader query", distinct_domains
        )
        results = _run_search(client, broader_query, max_results + 3)

    logger.info("Live search tool returned %d results for claim: %s", len(results), claim_text[:60])
    return results


def _run_search(client: TavilyClient, query: str, max_results: int) -> list[dict]:
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )
        return response.get("results", [])
    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return []


def _count_distinct_domains(results: list[dict]) -> int:
    domains = set()
    for r in results:
        url = r.get("url", "")
        parts = url.split("/")
        if len(parts) >= 3:
            domains.add(parts[2])
    return len(domains)


def format_search_context(results: list[dict]) -> tuple[str, list[str]]:
    """Format Tavily results into an evidence block string and a list of URLs.

    Returns:
        (context_block, evidence_links)
    """
    if not results:
        return "[LIVE SEARCH] No results found.", []

    lines = ["[LIVE SEARCH RESULTS]"]
    evidence_links: list[str] = []

    for r in results:
        url     = r.get("url", "")
        title   = r.get("title", "Untitled")
        content = (r.get("content") or r.get("snippet") or "")[:300]
        score   = r.get("score")

        score_str = f" (relevance: {score:.2f})" if score else ""
        lines.append(f"- [{title}]({url}){score_str}: {content}")

        if url:
            evidence_links.append(url)

    return "\n".join(lines), evidence_links
