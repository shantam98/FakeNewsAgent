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
    memory_results: Optional[MemoryQueryResponse]  # None until query_memory runs
    entity_context: list[dict]                     # from MemoryAgent.get_entity_context()

    # ── Routing ───────────────────────────────────────────────────────────────
    route: Optional[str]   # "cache" | "live_search"

    # ── Intermediate fact-check data ──────────────────────────────────────────
    retrieved_chunks: list[str]       # live search result + RAG context blocks
    sub_claims: list[str]             # SOTA: claim decomposition (unused in baseline)
    debate_transcript: Optional[str]  # SOTA: multi-agent debate (unused in baseline)

    # ── Cross-modal ────────────────────────────────────────────────────────────
    cross_modal_flag: bool
    cross_modal_explanation: Optional[str]
    clip_similarity_score: Optional[float]  # SOTA: CLIP scoring (unused in baseline)

    # ── Final output ──────────────────────────────────────────────────────────
    output: Optional[FactCheckOutput]


# Default values passed alongside FactCheckInput in graph.invoke()
INITIAL_STATE: dict = {
    "memory_results":          None,
    "entity_context":          [],
    "route":                   None,
    "retrieved_chunks":        [],
    "sub_claims":              [],
    "debate_transcript":       None,
    "cross_modal_flag":        False,
    "cross_modal_explanation": None,
    "clip_similarity_score":   None,
    "output":                  None,
}
