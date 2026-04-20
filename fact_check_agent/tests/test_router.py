"""Tests for the graph routing functions — no API keys required."""
from fact_check_agent.src.graph.router import (
    CACHE_CONFIDENCE_THRESHOLD,
    debate_check,
    freshness_router,
    retrieval_gate_router,
    router,
)
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

# ── freshness_router (T1) ────────────────────────────────────────────────────

def test_freshness_router_stale_when_true():
    assert freshness_router({"revalidation_needed": True}) == "stale"


def test_freshness_router_fresh_when_false():
    assert freshness_router({"revalidation_needed": False}) == "fresh"


def test_freshness_router_stale_when_none():
    """None is falsy — router must default to 'stale' (safe fallback)."""
    assert freshness_router({"revalidation_needed": None}) == "stale"


def test_freshness_router_stale_when_key_missing():
    assert freshness_router({}) == "stale"


# ── retrieval_gate_router (S2) ────────────────────────────────────────────────

def test_retrieval_gate_router_needed_when_true():
    assert retrieval_gate_router({"retrieval_gate_needed": True}) == "needed"


def test_retrieval_gate_router_skip_when_false():
    assert retrieval_gate_router({"retrieval_gate_needed": False}) == "skip"


def test_retrieval_gate_router_needed_when_key_missing():
    """Missing key defaults to needed (safe default — don't silently skip live search)."""
    assert retrieval_gate_router({}) == "needed"


# ── debate_check ──────────────────────────────────────────────────────────────

def test_debate_check_returns_skip_when_use_debate_false():
    """Baseline: use_debate=False → debate_check must always return 'skip'."""
    from unittest.mock import patch
    with patch("fact_check_agent.src.graph.router.settings") as ms:
        ms.use_debate = False
        assert debate_check({}) == "skip"


def test_debate_check_returns_skip_regardless_of_confidence_when_disabled():
    """Even low-confidence outputs should route to skip when debate is disabled."""
    from unittest.mock import MagicMock, patch
    from fact_check_agent.src.models.schemas import FactCheckOutput

    output = MagicMock(spec=FactCheckOutput)
    output.confidence_score = 50
    with patch("fact_check_agent.src.graph.router.settings") as ms:
        ms.use_debate = False
        assert debate_check({"output": output}) == "skip"


def test_debate_check_returns_debate_when_enabled_and_low_confidence():
    """When use_debate=True and confidence < threshold → should route to 'debate'."""
    from unittest.mock import MagicMock, patch
    from fact_check_agent.src.models.schemas import FactCheckOutput

    output = MagicMock(spec=FactCheckOutput)
    output.confidence_score = 45
    with patch("fact_check_agent.src.graph.router.settings") as ms:
        ms.use_debate = True
        ms.debate_confidence_threshold = 70
        assert debate_check({"output": output}) == "debate"


def test_debate_check_returns_skip_when_enabled_and_high_confidence():
    """When use_debate=True but confidence >= threshold → skip debate."""
    from unittest.mock import MagicMock, patch
    from fact_check_agent.src.models.schemas import FactCheckOutput

    output = MagicMock(spec=FactCheckOutput)
    output.confidence_score = 85
    with patch("fact_check_agent.src.graph.router.settings") as ms:
        ms.use_debate = True
        ms.debate_confidence_threshold = 70
        assert debate_check({"output": output}) == "skip"
