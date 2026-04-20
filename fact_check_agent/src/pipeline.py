"""Real pipeline entry point: PreprocessingOutput → list[FactCheckOutput].

Called after MemoryAgent.ingest_preprocessed() in the main data pipeline.
Converts each Claim in a PreprocessingOutput into a FactCheckInput, pre-fetches
the image caption once per article, then runs the LangGraph graph per claim.

Usage:
    from fact_check_agent.src.pipeline import run_fact_check
    from fact_check_agent.src.memory_client import get_memory

    memory  = get_memory()
    outputs = run_fact_check(preprocessing_output)
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from src._bootstrap import *  # noqa: F401,F403
from src.models.pipeline import PreprocessingOutput  # memory_agent model

from fact_check_agent.src.memory_client import get_memory
from fact_check_agent.src.graph.graph import build_graph
from fact_check_agent.src.models.schemas import EntityRef, FactCheckInput, FactCheckOutput

logger = logging.getLogger(__name__)

# Module-level compiled graph — built once, reused across calls
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(get_memory())
    return _graph


def claim_to_fact_check_input(
    output: PreprocessingOutput,
    claim_index: int,
    image_caption: Optional[str],
    image_url: Optional[str] = None,
) -> FactCheckInput:
    """Convert one Claim from a PreprocessingOutput into a typed FactCheckInput.

    Args:
        output:        PreprocessingOutput produced by PreprocessingAgent.
        claim_index:   Which claim in output.claims to convert.
        image_caption: Pre-fetched VLM caption string, or None.
        image_url:     Original image URL from ImageCaption (for SigLIP/vision check).
    """
    claim = output.claims[claim_index]
    return FactCheckInput(
        claim_id      = claim.claim_id,
        claim_text    = claim.claim_text,
        entities      = [
            EntityRef(
                entity_id   = e.entity_id,
                name        = e.name,
                entity_type = e.entity_type,
                sentiment   = e.sentiment,
            )
            for e in claim.entities
        ],
        source_url    = output.article.url,
        article_id    = claim.article_id,
        image_caption = image_caption,
        image_url     = image_url,
        timestamp     = claim.extracted_at,
    )


def run_fact_check(output: PreprocessingOutput) -> list[FactCheckOutput]:
    """Run the fact-check graph on every claim in a PreprocessingOutput.

    Pre-fetches the article's image caption once (shared across all claims),
    then invokes the LangGraph graph per claim.

    Returns one FactCheckOutput per claim in output.claims.
    """
    memory = get_memory()
    graph  = _get_graph()

    # Pre-fetch image caption once for the article (both text and URL)
    caption_result = memory.get_caption_by_article(output.article.article_id)
    image_caption: Optional[str] = None
    image_url: Optional[str] = None
    if caption_result.get("documents"):
        image_caption = caption_result["documents"][0]
    if caption_result.get("metadatas") and caption_result["metadatas"]:
        image_url = caption_result["metadatas"][0].get("image_url")

    results: list[FactCheckOutput] = []
    for i in range(len(output.claims)):
        fact_check_input = claim_to_fact_check_input(output, i, image_caption, image_url)
        logger.info(
            "Running fact-check for claim %d/%d: %s",
            i + 1, len(output.claims), fact_check_input.claim_id,
        )
        state = graph.invoke({"input": fact_check_input})
        fc_output: Optional[FactCheckOutput] = state.get("output")

        if fc_output:
            results.append(fc_output)
        else:
            logger.error("Graph returned no output for claim %s", fact_check_input.claim_id)

    return results
