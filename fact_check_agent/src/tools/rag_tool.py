"""RAG tool — retrieves similar claims and their verdicts from MemoryAgent.

This is a tool, not an agent: it makes direct database calls (ChromaDB vector
search + ChromaDB verdict lookup) and returns structured results. No LLM call,
no loops, no planning.
"""
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.agent import MemoryAgent  # memory_agent type hint only

logger = logging.getLogger(__name__)


def retrieve_similar_claims(claim_text: str, memory: "MemoryAgent", top_k: int = 5) -> list[dict]:
    """Query ChromaDB for semantically similar past claims and their verdicts.

    Returns a list of dicts with keys:
        claim_id, claim_text, verdict_label, verdict_confidence, distance, verified_at
    """
    raw = memory.search_similar_claims(claim_text, top_k=top_k)

    if not raw or not raw.get("ids") or not raw["ids"][0]:
        logger.debug("No similar claims found in memory for query: %s", claim_text[:60])
        return []

    ids       = raw["ids"][0]
    docs      = raw["documents"][0]
    distances = raw["distances"][0]

    results = []
    for i, claim_id in enumerate(ids):
        verdict_label      = None
        verdict_confidence = None
        verified_at        = None

        verdict_raw = memory.get_verdict_by_claim(claim_id)
        if verdict_raw.get("metadatas") and verdict_raw["metadatas"]:
            meta = verdict_raw["metadatas"][0]
            verdict_label      = meta.get("label")
            verdict_confidence = meta.get("confidence")
            raw_ts = meta.get("verified_at")
            if raw_ts:
                try:
                    verified_at = datetime.fromisoformat(raw_ts)
                    if verified_at.tzinfo is None:
                        verified_at = verified_at.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    verified_at = None

        results.append({
            "claim_id":           claim_id,
            "claim_text":         docs[i],
            "verdict_label":      verdict_label,
            "verdict_confidence": float(verdict_confidence) if verdict_confidence is not None else None,
            "distance":           float(distances[i]),
            "verified_at":        verified_at,
        })

    logger.debug("RAG tool retrieved %d similar claims", len(results))
    return results


def format_rag_context(similar_claims: list[dict]) -> str:
    """Format retrieved claims into a context block for the synthesis prompt."""
    if not similar_claims:
        return "[MEMORY] No similar claims found in memory."

    lines = ["[RETRIEVED EVIDENCE FROM MEMORY]"]
    for c in similar_claims:
        if c["verdict_label"]:
            conf_str = f"{c['verdict_confidence']:.0%}" if c["verdict_confidence"] else "?"
            verdict_str = f"{c['verdict_label']} ({conf_str} confidence)"
        else:
            verdict_str = "no prior verdict"
        lines.append(f'- Claim: "{c["claim_text"]}" | Prior verdict: {verdict_str}')

    return "\n".join(lines)
