"""Routing functions for the LangGraph conditional edges."""
from fact_check_agent.src.config import settings
from fact_check_agent.src.models.state import FactCheckState


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
