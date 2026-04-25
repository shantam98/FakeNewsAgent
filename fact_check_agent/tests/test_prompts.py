"""Tests for prompt templates — verify all placeholders are present."""
import re

from fact_check_agent.src.prompts import (
    CHUNK_RELEVANCE_PROMPT,
    CROSS_MODAL_PROMPT,
    DECOMPOSITION_PROMPT,
    IS_RETRIEVAL_NEEDED_PROMPT,
    JUDGE_PROMPT,
    SKEPTIC_PROMPT,
    SUPPORTER_PROMPT,
    VERDICT_SYNTHESIS_PROMPT,
)


def placeholders(template: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"\{(\w+)\}", template)}


def test_verdict_synthesis_has_required_placeholders():
    required = {"claim_text", "numbered_claims"}
    assert required.issubset(placeholders(VERDICT_SYNTHESIS_PROMPT))


def test_cross_modal_has_required_placeholders():
    required = {"claim_text", "image_caption"}
    assert required.issubset(placeholders(CROSS_MODAL_PROMPT))


def test_is_retrieval_needed_has_required_placeholders():
    assert "claim_text" in placeholders(IS_RETRIEVAL_NEEDED_PROMPT)


def test_chunk_relevance_has_required_placeholders():
    required = {"claim_text", "chunks_block"}
    assert required.issubset(placeholders(CHUNK_RELEVANCE_PROMPT))


def test_decomposition_has_required_placeholders():
    assert "claim_text" in placeholders(DECOMPOSITION_PROMPT)


def test_supporter_has_required_placeholders():
    required = {"claim_text", "numbered_claims", "neutral_scores_block"}
    assert required.issubset(placeholders(SUPPORTER_PROMPT))


def test_skeptic_has_required_placeholders():
    required = {"claim_text", "numbered_claims", "neutral_scores_block"}
    assert required.issubset(placeholders(SKEPTIC_PROMPT))


def test_judge_has_required_placeholders():
    required = {"claim_text", "numbered_claims", "neutral_scores_block",
                "supporter_adjustments", "skeptic_adjustments"}
    assert required.issubset(placeholders(JUDGE_PROMPT))


def test_all_prompts_are_non_empty():
    prompts = [
        VERDICT_SYNTHESIS_PROMPT,
        CROSS_MODAL_PROMPT,
        IS_RETRIEVAL_NEEDED_PROMPT,
        CHUNK_RELEVANCE_PROMPT,
        DECOMPOSITION_PROMPT,
        SUPPORTER_PROMPT,
        SKEPTIC_PROMPT,
        JUDGE_PROMPT,
    ]
    for prompt in prompts:
        assert isinstance(prompt, str) and len(prompt) > 0
