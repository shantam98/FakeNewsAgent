"""Retrieval reranking: Reciprocal Rank Fusion + optional cross-encoder.

Two-stage pipeline:
  1. RRF  — merges vector DB results and graph entity-claim results into one
            ranked list without needing score normalisation.
  2. Cross-encoder (optional, gated by USE_CROSS_ENCODER) — rescores each
            candidate against the query using a local sentence-transformers
            model for higher accuracy.
"""
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant — dampens high rank outliers


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "claim_id",
) -> list[dict]:
    """Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    Each list must be ordered best-first. Returns a deduplicated list ordered
    by descending RRF score. The original item dict is preserved; an rrf_score
    key is added.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = item[id_key]
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if key not in items:
                items[key] = item

    merged = sorted(items.values(), key=lambda x: scores[x[id_key]], reverse=True)
    for item in merged:
        item["rrf_score"] = scores[item[id_key]]
    return merged


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    logger.info("Loading cross-encoder model: %s", model_name)
    return CrossEncoder(model_name)


def cross_encoder_rerank(
    query: str,
    candidates: list[dict],
    model_name: str,
    top_k: int,
    text_key: str = "claim_text",
) -> list[dict]:
    """Rerank candidates using a cross-encoder model.

    Scores each (query, candidate_text) pair, returns top_k sorted by score.
    Falls back to the original order if the model fails.
    """
    if not candidates:
        return candidates

    try:
        model = _load_cross_encoder(model_name)
        pairs = [(query, c[text_key]) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(scores, candidates), key=lambda x: x[0], reverse=True
        )
        result = []
        for score, item in ranked[:top_k]:
            item = dict(item)
            item["cross_encoder_score"] = float(score)
            result.append(item)
        logger.debug(
            "Cross-encoder reranked %d → %d candidates", len(candidates), len(result)
        )
        return result
    except Exception as e:
        logger.error("Cross-encoder reranking failed: %s — returning original order", e)
        return candidates[:top_k]


def rerank_candidates(
    query: str,
    vector_results: list[dict],
    graph_results: list[dict],
    use_cross_encoder: bool,
    cross_encoder_model: str,
    top_k: int,
) -> list[dict]:
    """Full reranking pipeline: RRF merge then optional cross-encoder.

    Args:
        vector_results: Ranked list from ChromaDB similarity search.
        graph_results:  Ranked list from Neo4j entity-claim traversal.
        use_cross_encoder: If True, apply cross-encoder after RRF.
        top_k: Maximum number of results to return.

    Returns:
        Deduplicated, reranked list of claim dicts.
    """
    lists_to_merge = [l for l in [vector_results, graph_results] if l]

    if not lists_to_merge:
        return []

    if len(lists_to_merge) == 1:
        merged = lists_to_merge[0][:top_k]
    else:
        merged = reciprocal_rank_fusion(lists_to_merge)
        logger.info(
            "RRF merged %d vector + %d graph → %d unique candidates",
            len(vector_results), len(graph_results), len(merged),
        )

    if use_cross_encoder:
        return cross_encoder_rerank(query, merged, cross_encoder_model, top_k)

    return merged[:top_k]
