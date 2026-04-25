"""LangGraph state schema for the Fact-Check Agent graph."""
from typing import Optional, TypedDict

from fact_check_agent.src.models.schemas import (
    FactCheckInput,
    FactCheckOutput,
    MemoryQueryResponse,
)


class FactCheckState(TypedDict):
    # ── Input (required — set before graph.invoke()) ─────────────────────────
    input: FactCheckInput

    # ── Memory query results ──────────────────────────────────────────────────
    memory_results: Optional[MemoryQueryResponse]
    entity_context: list[dict]

    # ── Freshness-tagged memory context ──────────────────────────────────────
    fresh_context: list[dict]
    stale_context: list[dict]

    # ── Context claims (from context_claim_agent) ─────────────────────────────
    context_claims: list[dict]

    # ── Prefetched chunks (benchmark mode — Factify2 doc + OCR) ──────────────
    retrieved_chunks: list[str]

    # ── Neutral synthesis output (fed into Supporter + Skeptic) ──────────────
    neutral_degrees: list[float]
    neutral_reasoning: Optional[str]

    # ── Multi-agent debate ────────────────────────────────────────────────────
    debate_transcript: Optional[str]

    # ── Source credibility (from Reflection Agent) ────────────────────────────
    source_credibility: Optional[dict]

    # ── Cross-modal ───────────────────────────────────────────────────────────
    cross_modal_flag: bool
    cross_modal_explanation: Optional[str]
    clip_similarity_score: Optional[float]

    # ── Final output ──────────────────────────────────────────────────────────
    output: Optional[FactCheckOutput]


# Default values passed alongside FactCheckInput in graph.invoke()
INITIAL_STATE: dict = {
    "memory_results":          None,
    "entity_context":          [],
    "fresh_context":           [],
    "stale_context":           [],
    "context_claims":          [],
    "retrieved_chunks":        [],
    "neutral_degrees":         [],
    "neutral_reasoning":       None,
    "debate_transcript":       None,
    "source_credibility":      None,
    "cross_modal_flag":        False,
    "cross_modal_explanation": None,
    "clip_similarity_score":   None,
    "output":                  None,
}
