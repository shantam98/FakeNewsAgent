"""Tests for the live search tool — no Tavily API key required."""
from unittest.mock import MagicMock, patch

from fact_check_agent.src.tools.live_search_tool import (
    _count_distinct_domains,
    format_search_context,
    search_live,
)


# ── _count_distinct_domains ───────────────────────────────────────────────────

def test_count_distinct_domains_all_same():
    results = [
        {"url": "https://bbc.co.uk/news/1"},
        {"url": "https://bbc.co.uk/news/2"},
        {"url": "https://bbc.co.uk/news/3"},
    ]
    assert _count_distinct_domains(results) == 1


def test_count_distinct_domains_all_different():
    results = [
        {"url": "https://bbc.co.uk/news/1"},
        {"url": "https://reuters.com/article/1"},
        {"url": "https://apnews.com/story/1"},
    ]
    assert _count_distinct_domains(results) == 3


def test_count_distinct_domains_empty():
    assert _count_distinct_domains([]) == 0


def test_count_distinct_domains_missing_url():
    results = [{"url": ""}, {"url": "https://bbc.co.uk/1"}]
    assert _count_distinct_domains(results) == 1


# ── format_search_context ─────────────────────────────────────────────────────

def test_format_search_context_empty():
    context, links = format_search_context([])
    assert "No results" in context
    assert links == []


def test_format_search_context_returns_links():
    results = [
        {"url": "https://reuters.com/1", "title": "Reuters story", "content": "Some evidence."},
        {"url": "https://bbc.co.uk/1",   "title": "BBC story",     "content": "More evidence."},
    ]
    context, links = format_search_context(results)
    assert "reuters.com" in context
    assert "bbc.co.uk" in context
    assert "https://reuters.com/1" in links
    assert "https://bbc.co.uk/1" in links


def test_format_search_context_truncates_long_content():
    long_content = "x" * 500
    results = [{"url": "https://example.com", "title": "T", "content": long_content}]
    context, _ = format_search_context(results)
    assert len(context) < 500


def test_format_search_context_no_url_excluded_from_links():
    results = [{"url": "", "title": "No URL", "content": "Content"}]
    _, links = format_search_context(results)
    assert links == []


# ── search_live ───────────────────────────────────────────────────────────────

def make_tavily_result(urls):
    return {"results": [
        {"url": url, "title": f"Title {i}", "content": "Evidence text", "score": 0.9}
        for i, url in enumerate(urls)
    ]}


def test_search_live_returns_results():
    with patch("fact_check_agent.src.tools.live_search_tool.TavilyClient") as mock_cls:
        mock_cls.return_value.search.return_value = make_tavily_result([
            "https://reuters.com/1",
            "https://bbc.co.uk/1",
            "https://apnews.com/1",
        ])
        results = search_live("test claim", api_key="fake-key")

    assert len(results) == 3


def test_search_live_retries_when_too_few_domains():
    few_domains = make_tavily_result(["https://bbc.co.uk/1", "https://bbc.co.uk/2"])
    many_domains = make_tavily_result([
        "https://reuters.com/1", "https://bbc.co.uk/1", "https://apnews.com/1"
    ])

    with patch("fact_check_agent.src.tools.live_search_tool.TavilyClient") as mock_cls:
        mock_cls.return_value.search.side_effect = [few_domains, many_domains]
        results = search_live("test claim", api_key="fake-key")

    assert mock_cls.return_value.search.call_count == 2
    assert len(results) == 3


def test_search_live_handles_tavily_error():
    with patch("fact_check_agent.src.tools.live_search_tool.TavilyClient") as mock_cls:
        mock_cls.return_value.search.side_effect = Exception("API error")
        results = search_live("test claim", api_key="fake-key")

    assert results == []
