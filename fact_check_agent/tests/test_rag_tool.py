"""Tests for the RAG tool — no API keys or database required."""
from unittest.mock import MagicMock

from fact_check_agent.src.tools.rag_tool import format_rag_context, retrieve_similar_claims


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_memory(ids=None, docs=None, distances=None, verdict_meta=None):
    """Return a mock MemoryAgent with controllable responses."""
    memory = MagicMock()
    memory.search_similar_claims.return_value = {
        "ids":       [ids or []],
        "documents": [docs or []],
        "distances": [distances or []],
        "metadatas": [[]],
    }
    memory.get_verdict_by_claim.return_value = {
        "metadatas": [verdict_meta] if verdict_meta else []
    }
    return memory


# ── format_rag_context ────────────────────────────────────────────────────────

def test_format_rag_context_empty_returns_fallback():
    result = format_rag_context([])
    assert "No similar claims" in result


def test_format_rag_context_with_verdict():
    claims = [{
        "claim_id": "clm_1",
        "claim_text": "Vaccines cause autism",
        "verdict_label": "refuted",
        "verdict_confidence": 0.95,
        "distance": 0.12,
    }]
    result = format_rag_context(claims)
    assert "refuted" in result
    assert "95%" in result
    assert "Vaccines cause autism" in result


def test_format_rag_context_without_verdict():
    claims = [{
        "claim_id": "clm_2",
        "claim_text": "The sky is green",
        "verdict_label": None,
        "verdict_confidence": None,
        "distance": 0.30,
    }]
    result = format_rag_context(claims)
    assert "no prior verdict" in result
    assert "The sky is green" in result


def test_format_rag_context_multiple_claims():
    claims = [
        {"claim_id": "c1", "claim_text": "Claim A", "verdict_label": "supported",
         "verdict_confidence": 0.80, "distance": 0.1},
        {"claim_id": "c2", "claim_text": "Claim B", "verdict_label": "refuted",
         "verdict_confidence": 0.70, "distance": 0.2},
    ]
    result = format_rag_context(claims)
    assert "Claim A" in result
    assert "Claim B" in result
    assert "supported" in result
    assert "refuted" in result


# ── retrieve_similar_claims ───────────────────────────────────────────────────

def test_retrieve_returns_empty_when_no_memory():
    memory = make_memory()  # empty ids
    results = retrieve_similar_claims("any claim", memory)
    assert results == []


def test_retrieve_returns_claim_with_verdict():
    memory = make_memory(
        ids=["clm_abc"],
        docs=["Prior claim text"],
        distances=[0.15],
        verdict_meta={"label": "supported", "confidence": 0.88},
    )
    results = retrieve_similar_claims("some claim", memory)

    assert len(results) == 1
    assert results[0]["claim_id"] == "clm_abc"
    assert results[0]["claim_text"] == "Prior claim text"
    assert results[0]["verdict_label"] == "supported"
    assert results[0]["verdict_confidence"] == 0.88
    assert results[0]["distance"] == 0.15


def test_retrieve_returns_claim_without_verdict():
    memory = make_memory(
        ids=["clm_xyz"],
        docs=["Another claim"],
        distances=[0.25],
        verdict_meta=None,
    )
    results = retrieve_similar_claims("query", memory)

    assert len(results) == 1
    assert results[0]["verdict_label"] is None
    assert results[0]["verdict_confidence"] is None


def test_retrieve_respects_top_k():
    memory = make_memory(
        ids=["c1", "c2", "c3"],
        docs=["Doc 1", "Doc 2", "Doc 3"],
        distances=[0.1, 0.2, 0.3],
    )
    retrieve_similar_claims("query", memory, top_k=3)
    memory.search_similar_claims.assert_called_once_with("query", top_k=3)
