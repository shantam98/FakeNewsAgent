"""Frozen JSON interface contracts for the Fact-Check Agent.

These are the single source of truth for all agent handoffs.
Do not change field names without versioning and notifying downstream consumers.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Shared entity reference ───────────────────────────────────────────────────

class EntityRef(BaseModel):
    """A named entity attached to a claim.

    Mirrors MentionSentiment from memory_agent — kept as a separate type so
    fact_check_agent has no hard import dependency on memory_agent models.
    """

    entity_id: str
    name: str
    entity_type: str  # "person" | "organization" | "country" | "location" | "event" | "product"
    sentiment: str    # "positive" | "negative" | "neutral"


# ── Input contract: Preprocessing → Fact-Check Agent ─────────────────────────

class FactCheckInput(BaseModel):
    """Input handed to the LangGraph graph via graph.invoke({"input": ...}).

    How each field is populated:
    - Real pipeline: constructed by pipeline.py from a Claim + PreprocessingOutput.
    - Benchmark path: constructed by BenchmarkRecord.to_fact_check_input().
    """

    claim_id: str                     # e.g. "clm_abc123" | "liar_2635.json"
    claim_text: str
    entities: list[EntityRef]         # empty [] in benchmark path; filled from Claim in real path
    source_url: str
    article_id: str                   # used to correlate with memory store
    image_caption: Optional[str] = None   # pre-fetched VLM caption; None if no image
    image_url: Optional[str] = None       # raw image URL or base64 data URI; used by cross-modal check
    timestamp: datetime
    prefetched_chunks: list[str] = Field(default_factory=list)  # pre-fetched evidence; skips live_search when non-empty


# ── Output contract: Fact-Check Agent → Frontend / Memory ────────────────────

class FactCheckOutput(BaseModel):
    """Output emitted by emit_output node and written to MemoryAgent."""

    verdict_id: str
    claim_id: str
    verdict: str                      # "supported" | "refuted" | "misleading"
    confidence_score: int = Field(ge=0, le=100)   # 0–100 int for API; stored as float/100 in memory
    evidence_links: list[str]         # source URLs supporting the verdict
    reasoning: str                    # chain-of-thought explanation
    bias_score: float = Field(ge=0.0, le=1.0)
    cross_modal_flag: bool = False
    cross_modal_explanation: Optional[str] = None  # one sentence; None if no image or no conflict
    last_verified_at: Optional[datetime] = None    # populated on cache hits; None on live-search path
    revalidation_needed: bool = False              # True if freshness_check decided live re-check is required


# ── Memory query types ────────────────────────────────────────────────────────

class SimilarClaim(BaseModel):
    claim_id: str
    claim_text: str
    verdict_label: Optional[str] = None       # None if claim has no verdict yet
    verdict_confidence: Optional[float] = None
    distance: float                            # ChromaDB cosine distance; lower = more similar
    verified_at: Optional[datetime] = None    # when this verdict was last written to memory


class MemoryQueryRequest(BaseModel):
    claim_text: str
    top_k: int = 5


class MemoryQueryResponse(BaseModel):
    results: list[SimilarClaim]
    max_confidence: float = 0.0   # highest verdict confidence among results; 0.0 if none
