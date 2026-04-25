"""LangGraph state schema for the Fact-Check Agent graph."""
from datetime import datetime
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
    memory_results: Optional[MemoryQueryResponse]  # None until query_memory runs
    entity_context: list[dict]                     # from MemoryAgent.get_entity_context()

    # ── Freshness-tagged memory context ──────────────────────────────────────
    fresh_context: list[dict]   # SimilarClaim dicts tagged as still-current
    stale_context: list[dict]   # SimilarClaim dicts tagged as outdated

    # ── Context claims (from context_claim_agent) ─────────────────────────────
    # Each dict: {type, question, content, verdict, confidence, source}
    context_claims: list[dict]

    # ── Prefetched chunks (benchmark mode — Factify2 doc + OCR) ──────────────
    retrieved_chunks: list[str]

    # ── Claim decomposition (S3) ──────────────────────────────────────────────
    sub_claims: list[str]

    # ── Multi-agent debate (S4) ───────────────────────────────────────────────
    debate_transcript: Optional[str]

    # ── Source credibility (from Reflection Agent) ────────────────────────────
    source_credibility: Optional[dict]  # keys: credibility_mean, bias_mean, bias_std, sample_count

    # ── Cross-modal ────────────────────────────────────────────────────────────
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
    "sub_claims":              [],
    "debate_transcript":       None,
    "source_credibility":      None,
    "cross_modal_flag":        False,
    "cross_modal_explanation": None,
    "clip_similarity_score":   None,
    "output":                  None,
}
