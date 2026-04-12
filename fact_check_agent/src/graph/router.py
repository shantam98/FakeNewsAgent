"""Routing functions for the LangGraph conditional edges."""
from fact_check_agent.src.models.state import FactCheckState

CACHE_CONFIDENCE_THRESHOLD = 0.80


def router(state: FactCheckState) -> str:
    """Route to 'cache' if memory has a high-confidence prior verdict, else 'live_search'."""
    memory_results = state.get("memory_results")
    if memory_results and memory_results.max_confidence >= CACHE_CONFIDENCE_THRESHOLD:
        return "cache"
    return "live_search"


def debate_check(state: FactCheckState) -> str:
    """Decide whether to trigger multi-agent debate.

    Baseline: always skips debate.
    SOTA: enable by returning "debate" when 35 < confidence_score < 65.
    """
    # SOTA gate — uncomment to enable multi-agent debate
    # output = state.get("output")
    # if output and 35 < output.confidence_score < 65:
    #     return "debate"
    return "skip"
