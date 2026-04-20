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

    # ── Routing ───────────────────────────────────────────────────────────────
    route: Optional[str]               # "cache" | "live_search"
    revalidation_needed: Optional[bool]  # set by freshness_check; True → re-run live search
    retrieval_gate_needed: Optional[bool]  # S2: set by retrieval_gate; False → skip Tavily

    # ── Intermediate fact-check data ──────────────────────────────────────────
    retrieved_chunks: list[str]       # live search result + RAG context blocks
    sub_claims: list[str]             # SOTA: claim decomposition (unused in baseline)
    debate_transcript: Optional[str]  # SOTA: multi-agent debate (unused in baseline)

    # ── Source credibility (from Reflection Agent) ────────────────────────────
    source_credibility: Optional[dict]  # keys: credibility_mean, bias_mean, bias_std, sample_count

    # ── Cross-modal ────────────────────────────────────────────────────────────
    cross_modal_flag: bool
    cross_modal_explanation: Optional[str]
    clip_similarity_score: Optional[float]  # SOTA: CLIP scoring (unused in baseline)

    # ── Freshness ─────────────────────────────────────────────────────────────
    last_verified_at: Optional[datetime]   # from best cache hit; None on live-search path

    # ── Final output ──────────────────────────────────────────────────────────
    output: Optional[FactCheckOutput]


# Default values passed alongside FactCheckInput in graph.invoke()
INITIAL_STATE: dict = {
    "memory_results":          None,
    "entity_context":          [],
    "route":                   None,
    "revalidation_needed":     None,
    "retrieval_gate_needed":   None,
    "retrieved_chunks":        [],
    "sub_claims":              [],
    "debate_transcript":       None,
    "source_credibility":      None,
    "cross_modal_flag":        False,
    "cross_modal_explanation": None,
    "clip_similarity_score":   None,
    "last_verified_at":        None,
    "output":                  None,
}
