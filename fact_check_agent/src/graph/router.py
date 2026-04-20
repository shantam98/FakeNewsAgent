"""Routing functions for the LangGraph conditional edges."""
from fact_check_agent.src.config import settings
from fact_check_agent.src.models.state import FactCheckState

CACHE_CONFIDENCE_THRESHOLD = 0.80


def router(state: FactCheckState) -> str:
    """Route to 'cache' if memory has a high-confidence prior verdict, else 'live_search'."""
    memory_results = state.get("memory_results")
    if memory_results and memory_results.max_confidence >= CACHE_CONFIDENCE_THRESHOLD:
        return "cache"
    return "live_search"


def freshness_router(state: FactCheckState) -> str:
    """After a cache hit, decide whether to serve the cached verdict or re-verify.

    Returns:
        "fresh"  → use cached verdict; skip live search (return_cached → synthesize)
        "stale"  → run live search before synthesizing (live_search → rag_retrieval → synthesize)
    """
    if state.get("revalidation_needed") is False:
        return "fresh"
    return "stale"  # True or None → stale (safe default)


def retrieval_gate_router(state: FactCheckState) -> str:
    """S2: After retrieval_gate node, route to live_search or skip directly to rag_retrieval.

    Returns:
        "needed" → proceed to Tavily live search
        "skip"   → skip Tavily; use memory context only (→ rag_retrieval)
    """
    if state.get("retrieval_gate_needed", True):
        return "needed"
    return "skip"


def debate_check(state: FactCheckState) -> str:
    """Decide whether to trigger multi-agent debate.

    Gated by settings.use_debate. When enabled, routes low-confidence verdicts
    through an advocate/arbiter debate loop before cross-modal check.
    """
    if settings.use_debate:
        output = state.get("output")
        if output and output.confidence_score < settings.debate_confidence_threshold:
            return "debate"
    return "skip"
