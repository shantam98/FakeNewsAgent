"""Reflection Agent — maintains per-source, topic-conditioned credibility signals.

Runs in two directions after each fact-check:

  READ  (query_source_credibility)
        Called by query_memory node BEFORE verdict synthesis.
        Fetches k nearest (source, topic) observations from ChromaDB and returns
        weighted statistics: credibility_mean, bias_mean, bias_std.
        These are injected into the synthesis prompt as context.

  WRITE (update_source_credibility)
        Called by write_memory node AFTER verdict synthesis.
        Appends one new observation (source_id, topic_embedding, credibility, bias)
        to the source_credibility collection. Always inserts — never upserts.

Design:
  - No LLM calls. No loops. No planning. This is a tool wrapped in an "agent" name
    because it accumulates system-level knowledge across many verdicts.
  - Topic embedding = embedding of the claim text. Modern embeddings cluster
    semantically similar topics without explicit normalisation.
  - Credibility signal is derived from the verdict: a source is credible when its
    claims are supported with high confidence, and not credible when its claims are
    refuted with high confidence.
  - Weighted k-NN aggregation at query time (distance-weighted mean and variance)
    avoids the temporal-ordering requirement of EMA and naturally handles the
    topic-conditional structure.
"""
import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from src.memory.agent import MemoryAgent  # memory_agent — type hint only

logger = logging.getLogger(__name__)

_DEFAULT_K = 20          # neighbours retrieved when querying
_MIN_SAMPLES = 2         # minimum observations to return a meaningful estimate


# ── Helpers ───────────────────────────────────────────────────────────────────

def source_id_from_url(source_url: str) -> str:
    """Derive source_id from a URL — mirrors preprocessing/agent.py convention.

    Example: "https://bbc.co.uk/news/1" → "src_bbc_co_uk"
    """
    domain = urlparse(source_url).netloc or source_url
    return f"src_{domain.replace('.', '_')}"


def credibility_signal(verdict_label: str, confidence_score: int) -> float:
    """Map a verdict to a per-source credibility observation in [0, 1].

    The credibility of a source reflects whether its published claims are true:
      - supported with confidence c  → source made a truthful claim  → credibility = c/100
      - refuted   with confidence c  → source made a false claim      → credibility = 1 - c/100
      - misleading                   → ambiguous signal               → 0.5

    A source that consistently publishes supported claims with high confidence
    will accumulate credibility_mean → 1.0 over time.
    A source that consistently publishes refuted claims will trend toward 0.0.
    """
    c = confidence_score / 100.0
    if verdict_label == "supported":
        return c
    if verdict_label == "refuted":
        return 1.0 - c
    return 0.5  # misleading — neutral signal


# ── Read path ─────────────────────────────────────────────────────────────────

def query_source_credibility(
    claim_text: str,
    source_url: str,
    memory: "MemoryAgent",
    k: int = _DEFAULT_K,
) -> dict:
    """Return weighted credibility/bias statistics for (source, topic).

    Returns a dict with keys:
        credibility_mean  float [0, 1]   — weighted mean source credibility
        bias_mean         float [0, 1]   — weighted mean bias score
        bias_std          float [0, 1]   — weighted standard deviation of bias
                                           (high std = inconsistent source)
        sample_count      int            — number of observations retrieved

    Returns None-filled stats if fewer than _MIN_SAMPLES observations exist.
    The None values signal to the caller that there is insufficient history —
    the synthesis prompt will say "no prior data" rather than fabricating a score.
    """
    source_id = source_id_from_url(source_url)

    try:
        results = memory.query_source_credibility(
            claim_text=claim_text,
            source_id=source_id,
            k=k,
        )
    except Exception as exc:
        logger.warning("query_source_credibility failed for %s: %s", source_id, exc)
        return {"credibility_mean": None, "bias_mean": None, "bias_std": None, "sample_count": 0}

    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if len(metadatas) < _MIN_SAMPLES:
        return {
            "credibility_mean": None,
            "bias_mean":        None,
            "bias_std":         None,
            "sample_count":     len(metadatas),
        }

    # Distance-weighted aggregation — closer topic = higher weight
    weights = [1.0 / (d + 1e-6) for d in distances]
    W = sum(weights)

    cred_mean = sum(w * m["credibility"] for w, m in zip(weights, metadatas)) / W
    bias_mean = sum(w * m["bias"]        for w, m in zip(weights, metadatas)) / W
    bias_var  = sum(w * (m["bias"] - bias_mean) ** 2 for w, m in zip(weights, metadatas)) / W
    bias_std  = math.sqrt(bias_var)

    logger.info(
        "source_credibility: source=%s n=%d cred=%.2f bias=%.2f±%.2f",
        source_id, len(metadatas), cred_mean, bias_mean, bias_std,
    )

    return {
        "credibility_mean": round(cred_mean, 4),
        "bias_mean":        round(bias_mean, 4),
        "bias_std":         round(bias_std, 4),
        "sample_count":     len(metadatas),
    }


# ── Write path ────────────────────────────────────────────────────────────────

def update_source_credibility(
    claim_text: str,
    source_url: str,
    verdict_id: str,
    verdict_label: str,
    confidence_score: int,
    bias_score: float,
    memory: "MemoryAgent",
) -> None:
    """Append one new (source, topic, credibility, bias) observation after a verdict.

    Called by write_memory node. Always inserts — never overwrites. The full
    observation history is preserved so query-time aggregation can weight and
    average across all past verdicts for this (source, topic) region.
    """
    source_id   = source_id_from_url(source_url)
    credibility = credibility_signal(verdict_label, confidence_score)
    point_id    = f"sc_{verdict_id}"
    created_at  = datetime.now(timezone.utc).isoformat()

    try:
        memory.add_source_credibility_point(
            point_id      = point_id,
            claim_text    = claim_text,
            topic_text    = claim_text,
            source_id     = source_id,
            credibility   = credibility,
            bias          = bias_score,
            verdict_label = verdict_label,
            verdict_id    = verdict_id,
            created_at    = created_at,
        )
        logger.info(
            "update_source_credibility: source=%s cred=%.2f bias=%.2f label=%s",
            source_id, credibility, bias_score, verdict_label,
        )
    except Exception as exc:
        # Non-fatal — verdict is already written; reflection failure should not
        # block the pipeline.
        logger.error("update_source_credibility failed for %s: %s", source_id, exc)
