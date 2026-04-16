"""Tests for the graph routing functions — no API keys required."""
from fact_check_agent.src.graph.router import CACHE_CONFIDENCE_THRESHOLD, debate_check, router
from fact_check_agent.src.models.schemas import MemoryQueryResponse, SimilarClaim


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(max_confidence=None):
    """Return a minimal FactCheckState-like dict for router tests."""
    if max_confidence is None:
        memory_results = None
    else:
        memory_results = MemoryQueryResponse(
            results=[
                SimilarClaim(
                    claim_id="c1",
                    claim_text="Prior claim",
                    verdict_label="supported",
                    verdict_confidence=max_confidence,
                    distance=0.1,
                )
            ],
            max_confidence=max_confidence,
        )
    return {"memory_results": memory_results}


# ── router ────────────────────────────────────────────────────────────────────

def test_router_returns_cache_when_confidence_above_threshold():
    state = make_state(max_confidence=CACHE_CONFIDENCE_THRESHOLD)
    assert router(state) == "cache"


def test_router_returns_cache_strictly_above_threshold():
    state = make_state(max_confidence=CACHE_CONFIDENCE_THRESHOLD + 0.01)
    assert router(state) == "cache"


def test_router_returns_live_search_when_confidence_below_threshold():
    state = make_state(max_confidence=CACHE_CONFIDENCE_THRESHOLD - 0.01)
    assert router(state) == "live_search"


def test_router_returns_live_search_when_no_memory():
    state = make_state(max_confidence=None)
    assert router(state) == "live_search"


def test_router_returns_live_search_when_zero_confidence():
    state = make_state(max_confidence=0.0)
    assert router(state) == "live_search"


def test_cache_confidence_threshold_is_0_80():
    """Sanity-check the threshold value so regressions are obvious."""
    assert CACHE_CONFIDENCE_THRESHOLD == 0.80


# ── debate_check ──────────────────────────────────────────────────────────────

def test_debate_check_always_returns_skip():
    """Baseline: debate is disabled — debate_check must always return 'skip'."""
    assert debate_check({}) == "skip"


def test_debate_check_returns_skip_regardless_of_confidence():
    """Even low-confidence outputs should route to skip in the baseline."""
    from unittest.mock import MagicMock
    from fact_check_agent.src.models.schemas import FactCheckOutput

    output = MagicMock(spec=FactCheckOutput)
    output.confidence_score = 50  # within the SOTA debate range (35–65)
    state = {"output": output}
    assert debate_check(state) == "skip"
