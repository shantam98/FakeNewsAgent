"""Tests for prompt templates — verify all placeholders are present.

These tests catch accidental deletion or renaming of prompt variables that
would cause KeyError at runtime when .format() is called.
"""
import re

from fact_check_agent.src.prompts import (
    ADVOCATE_PROMPT,
    ARBITER_PROMPT,
    CHUNK_RELEVANCE_PROMPT,
    CROSS_MODAL_PROMPT,
    DECOMPOSITION_PROMPT,
    IS_RETRIEVAL_NEEDED_PROMPT,
    VERDICT_SYNTHESIS_PROMPT,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def placeholders(template: str) -> set[str]:
    """Return the set of {placeholder} names in a format string."""
    return {m.group(1) for m in re.finditer(r"\{(\w+)\}", template)}


# ── VERDICT_SYNTHESIS_PROMPT ──────────────────────────────────────────────────

def test_verdict_synthesis_has_required_placeholders():
    required = {"claim_text", "evidence_block", "source_credibility_note"}
    assert required.issubset(placeholders(VERDICT_SYNTHESIS_PROMPT))


def test_verdict_synthesis_format_succeeds():
    result = VERDICT_SYNTHESIS_PROMPT.format(
        claim_text="Test claim",
        evidence_block="Some evidence",
        source_credibility_note="Source: example.com",
    )
    assert "Test claim" in result
    assert "Some evidence" in result


# ── CROSS_MODAL_PROMPT ────────────────────────────────────────────────────────

def test_cross_modal_has_required_placeholders():
    required = {"claim_text", "image_caption"}
    assert required.issubset(placeholders(CROSS_MODAL_PROMPT))


def test_cross_modal_format_succeeds():
    result = CROSS_MODAL_PROMPT.format(
        claim_text="Protest was peaceful",
        image_caption="Police disperse crowd",
    )
    assert "Protest was peaceful" in result
    assert "Police disperse crowd" in result


# ── IS_RETRIEVAL_NEEDED_PROMPT ────────────────────────────────────────────────

def test_is_retrieval_needed_has_required_placeholders():
    assert "claim_text" in placeholders(IS_RETRIEVAL_NEEDED_PROMPT)


# ── CHUNK_RELEVANCE_PROMPT ────────────────────────────────────────────────────

def test_chunk_relevance_has_required_placeholders():
    required = {"claim_text", "chunks_block"}
    assert required.issubset(placeholders(CHUNK_RELEVANCE_PROMPT))


# ── DECOMPOSITION_PROMPT ──────────────────────────────────────────────────────

def test_decomposition_has_required_placeholders():
    assert "claim_text" in placeholders(DECOMPOSITION_PROMPT)


# ── ADVOCATE_PROMPT ───────────────────────────────────────────────────────────

def test_advocate_has_required_placeholders():
    required = {"position", "position_adj", "claim_text", "evidence_block"}
    assert required.issubset(placeholders(ADVOCATE_PROMPT))


# ── ARBITER_PROMPT ────────────────────────────────────────────────────────────

def test_arbiter_has_required_placeholders():
    required = {"claim_text", "argument_for", "argument_against"}
    assert required.issubset(placeholders(ARBITER_PROMPT))


def test_arbiter_format_succeeds():
    result = ARBITER_PROMPT.format(
        claim_text="Test claim",
        argument_for="Evidence A",
        argument_against="Evidence B",
    )
    assert "Test claim" in result
    assert "Evidence A" in result


# ── All prompts are non-empty strings ─────────────────────────────────────────

def test_all_prompts_are_non_empty():
    prompts = [
        VERDICT_SYNTHESIS_PROMPT,
        CROSS_MODAL_PROMPT,
        IS_RETRIEVAL_NEEDED_PROMPT,
        CHUNK_RELEVANCE_PROMPT,
        DECOMPOSITION_PROMPT,
        ADVOCATE_PROMPT,
        ARBITER_PROMPT,
    ]
    for prompt in prompts:
        assert isinstance(prompt, str) and len(prompt) > 0
