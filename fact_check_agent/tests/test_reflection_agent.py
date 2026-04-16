"""Tests for the Reflection Agent — no ChromaDB or OpenAI API keys required.

Covers:
  - source_id_from_url: URL → source_id derivation
  - credibility_signal: verdict + confidence → [0,1] credibility observation
  - query_source_credibility: k-NN weighted aggregation (mean, bias_mean, bias_std)
  - update_source_credibility: delegates correctly to MemoryAgent
"""
from unittest.mock import MagicMock, call, patch

from fact_check_agent.src.agents.reflection_agent import (
    _MIN_SAMPLES,
    credibility_signal,
    query_source_credibility,
    source_id_from_url,
    update_source_credibility,
)


# ── source_id_from_url ────────────────────────────────────────────────────────

def test_source_id_from_standard_url():
    assert source_id_from_url("https://bbc.co.uk/news/1") == "src_bbc_co_uk"


def test_source_id_from_url_strips_path():
    """Only the domain should be in the ID — not the path."""
    sid = source_id_from_url("https://reuters.com/article/long/path?q=1")
    assert sid == "src_reuters_com"
    assert "/article" not in sid


def test_source_id_from_url_replaces_dots():
    """Dots in the domain are replaced with underscores."""
    sid = source_id_from_url("https://apnews.com/article/1")
    assert "." not in sid.replace("src_", "")


def test_source_id_same_for_same_domain():
    """Two different URLs from the same domain must produce the same source_id."""
    a = source_id_from_url("https://bbc.co.uk/news/1")
    b = source_id_from_url("https://bbc.co.uk/news/2")
    assert a == b


# ── credibility_signal ────────────────────────────────────────────────────────

def test_credibility_signal_supported_high_confidence():
    assert credibility_signal("supported", 90) == 0.90


def test_credibility_signal_supported_low_confidence():
    assert credibility_signal("supported", 40) == 0.40


def test_credibility_signal_refuted_high_confidence():
    """Refuted with 90% confidence → source was dishonest → credibility = 0.10."""
    assert abs(credibility_signal("refuted", 90) - 0.10) < 1e-9


def test_credibility_signal_refuted_low_confidence():
    """Refuted with 30% confidence → ambiguous → credibility = 0.70."""
    assert abs(credibility_signal("refuted", 30) - 0.70) < 1e-9


def test_credibility_signal_misleading_is_neutral():
    """Misleading verdict always returns the neutral 0.5 regardless of confidence."""
    assert credibility_signal("misleading", 0)   == 0.5
    assert credibility_signal("misleading", 100) == 0.5
    assert credibility_signal("misleading", 55)  == 0.5


def test_credibility_signal_boundary_values():
    """confidence_score=0 and confidence_score=100 must not produce out-of-range values."""
    assert 0.0 <= credibility_signal("supported", 0)   <= 1.0
    assert 0.0 <= credibility_signal("supported", 100) <= 1.0
    assert 0.0 <= credibility_signal("refuted",   0)   <= 1.0
    assert 0.0 <= credibility_signal("refuted",   100) <= 1.0


# ── query_source_credibility ──────────────────────────────────────────────────

def make_chroma_results(distances, credibilities, biases):
    """Build a ChromaDB-style query result dict."""
    metadatas = [
        {"credibility": c, "bias": b, "source_id": "src_test_com",
         "topic_text": "test", "verdict_label": "supported",
         "verdict_id": f"vrd_{i}", "created_at": "2024-01-01T00:00:00+00:00"}
        for i, (c, b) in enumerate(zip(credibilities, biases))
    ]
    return {"distances": [distances], "metadatas": [metadatas]}


def test_query_returns_weighted_mean():
    """credibility_mean must be the distance-weighted average of observations."""
    memory = MagicMock()
    # Two equal-distance observations: credibility 1.0 and 0.0 → mean should be 0.5
    memory.query_source_credibility.return_value = make_chroma_results(
        distances=[0.1, 0.1],
        credibilities=[1.0, 0.0],
        biases=[0.2, 0.8],
    )

    result = query_source_credibility("any claim", "https://test.com", memory)

    assert abs(result["credibility_mean"] - 0.5) < 1e-3
    assert result["sample_count"] == 2


def test_query_closer_neighbour_weighted_more():
    """Observation at distance 0.1 should dominate over one at distance 0.9."""
    memory = MagicMock()
    memory.query_source_credibility.return_value = make_chroma_results(
        distances=[0.1, 0.9],
        credibilities=[1.0, 0.0],  # close = high credibility, far = low
        biases=[0.1, 0.9],
    )

    result = query_source_credibility("any claim", "https://test.com", memory)

    # Closer neighbour dominates — mean should be above 0.5
    assert result["credibility_mean"] > 0.5


def test_query_bias_std_high_when_inconsistent():
    """High bias variance should be reflected in bias_std."""
    memory = MagicMock()
    memory.query_source_credibility.return_value = make_chroma_results(
        distances=[0.1, 0.1],
        credibilities=[0.5, 0.5],
        biases=[0.0, 1.0],  # extreme values → high std
    )

    result = query_source_credibility("any claim", "https://test.com", memory)

    assert result["bias_std"] > 0.3


def test_query_bias_std_low_when_consistent():
    """Low bias variance should produce a near-zero bias_std."""
    memory = MagicMock()
    memory.query_source_credibility.return_value = make_chroma_results(
        distances=[0.1, 0.1, 0.1],
        credibilities=[0.8, 0.8, 0.8],
        biases=[0.2, 0.2, 0.2],  # all the same → std = 0
    )

    result = query_source_credibility("any claim", "https://test.com", memory)

    assert result["bias_std"] < 1e-6


def test_query_insufficient_samples_returns_none_stats():
    """Fewer than _MIN_SAMPLES observations → all stat fields are None."""
    memory = MagicMock()
    memory.query_source_credibility.return_value = make_chroma_results(
        distances=[0.1],
        credibilities=[0.9],
        biases=[0.1],
    )

    result = query_source_credibility("any claim", "https://test.com", memory, k=20)

    assert result["credibility_mean"] is None
    assert result["bias_mean"]        is None
    assert result["bias_std"]         is None
    assert result["sample_count"]     == 1


def test_query_empty_results_returns_none_stats():
    """Zero observations → all stat fields are None, sample_count = 0."""
    memory = MagicMock()
    memory.query_source_credibility.return_value = {"distances": [[]], "metadatas": [[]]}

    result = query_source_credibility("any claim", "https://test.com", memory)

    assert result["credibility_mean"] is None
    assert result["sample_count"]     == 0


def test_query_memory_failure_returns_none_stats():
    """If MemoryAgent raises, degrade gracefully — don't crash the pipeline."""
    memory = MagicMock()
    memory.query_source_credibility.side_effect = Exception("ChromaDB unavailable")

    result = query_source_credibility("any claim", "https://test.com", memory)

    assert result["credibility_mean"] is None
    assert result["sample_count"]     == 0


def test_query_uses_correct_source_id():
    """The source_id passed to MemoryAgent must match source_id_from_url output."""
    memory = MagicMock()
    memory.query_source_credibility.return_value = {"distances": [[]], "metadatas": [[]]}

    query_source_credibility("any claim", "https://bbc.co.uk/news/1", memory)

    _call_kwargs = memory.query_source_credibility.call_args
    assert _call_kwargs.kwargs["source_id"] == "src_bbc_co_uk"


# ── update_source_credibility ─────────────────────────────────────────────────

def test_update_calls_memory_add():
    """update_source_credibility must call memory.add_source_credibility_point once."""
    memory = MagicMock()
    memory.add_source_credibility_point.return_value = None

    update_source_credibility(
        claim_text      = "Test claim.",
        source_url      = "https://example.com/article",
        verdict_id      = "vrd_abc123",
        verdict_label   = "supported",
        confidence_score= 80,
        bias_score      = 0.3,
        memory          = memory,
    )

    memory.add_source_credibility_point.assert_called_once()


def test_update_point_id_format():
    """point_id must be 'sc_{verdict_id}' to prevent collision with other collections."""
    memory = MagicMock()
    memory.add_source_credibility_point.return_value = None

    update_source_credibility(
        claim_text="Claim.", source_url="https://example.com",
        verdict_id="vrd_xyz789", verdict_label="refuted",
        confidence_score=70, bias_score=0.6, memory=memory,
    )

    call_kwargs = memory.add_source_credibility_point.call_args.kwargs
    assert call_kwargs["point_id"] == "sc_vrd_xyz789"


def test_update_credibility_signal_applied():
    """credibility written to memory must use credibility_signal() — not raw confidence."""
    memory = MagicMock()
    memory.add_source_credibility_point.return_value = None

    update_source_credibility(
        claim_text="Claim.", source_url="https://example.com",
        verdict_id="vrd_1", verdict_label="refuted",
        confidence_score=80, bias_score=0.5, memory=memory,
    )

    call_kwargs = memory.add_source_credibility_point.call_args.kwargs
    # refuted with 80 confidence → credibility = 1 - 0.80 = 0.20
    assert abs(call_kwargs["credibility"] - 0.20) < 1e-9


def test_update_memory_failure_does_not_raise():
    """If MemoryAgent raises, update_source_credibility must not propagate the exception."""
    memory = MagicMock()
    memory.add_source_credibility_point.side_effect = Exception("write failed")

    # Should not raise
    update_source_credibility(
        claim_text="Claim.", source_url="https://example.com",
        verdict_id="vrd_1", verdict_label="supported",
        confidence_score=80, bias_score=0.3, memory=memory,
    )
