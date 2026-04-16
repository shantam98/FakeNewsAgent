"""Tests for the cross-modal tool — no OpenAI API key required."""
import json
from unittest.mock import MagicMock, patch

from fact_check_agent.src.tools.cross_modal_tool import check_cross_modal


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_openai_response(conflict: bool, explanation=None):
    """Return a mock OpenAI ChatCompletion with the given cross-modal result."""
    content = json.dumps({"conflict": conflict, "explanation": explanation})
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ── No image caption ─────────────────────────────────────────────────────────

def test_no_image_caption_returns_no_flag():
    """When there is no image caption, skip the check entirely."""
    result = check_cross_modal(
        claim_text="Some claim",
        image_caption=None,
        api_key="fake-key",
        model="gpt-4o",
    )
    assert result["flag"] is False
    assert result["explanation"] is None
    assert result["clip_score"] is None


def test_empty_image_caption_returns_no_flag():
    result = check_cross_modal(
        claim_text="Some claim",
        image_caption="",
        api_key="fake-key",
        model="gpt-4o",
    )
    assert result["flag"] is False


# ── LLM says no conflict ──────────────────────────────────────────────────────

def test_no_conflict_llm_returns_false_flag():
    with patch("fact_check_agent.src.tools.cross_modal_tool.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_openai_response(
            conflict=False
        )
        result = check_cross_modal(
            claim_text="Vaccines are safe.",
            image_caption="Doctor administering vaccine to patient.",
            api_key="fake-key",
            model="gpt-4o",
        )

    assert result["flag"] is False
    assert result["clip_score"] is None


# ── LLM says conflict ─────────────────────────────────────────────────────────

def test_conflict_llm_returns_true_flag():
    with patch("fact_check_agent.src.tools.cross_modal_tool.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_openai_response(
            conflict=True,
            explanation="Image shows protest but claim says peaceful gathering.",
        )
        result = check_cross_modal(
            claim_text="The rally was peaceful.",
            image_caption="Police disperse violent crowd.",
            api_key="fake-key",
            model="gpt-4o",
        )

    assert result["flag"] is True
    assert result["explanation"] is not None


# ── LLM API failure ───────────────────────────────────────────────────────────

def test_llm_failure_returns_no_flag():
    """If OpenAI call fails, the check should degrade gracefully to flag=False."""
    with patch("fact_check_agent.src.tools.cross_modal_tool.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = Exception("API error")
        result = check_cross_modal(
            claim_text="Some claim",
            image_caption="Some caption",
            api_key="fake-key",
            model="gpt-4o",
        )

    assert result["flag"] is False


# ── CLIP gate is off by default ───────────────────────────────────────────────

def test_clip_score_is_none_when_disabled():
    """ENABLE_CLIP=False by default — clip_score should always be None."""
    with patch("fact_check_agent.src.tools.cross_modal_tool.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = make_openai_response(
            conflict=False
        )
        result = check_cross_modal(
            claim_text="Claim text",
            image_caption="Some caption",
            api_key="fake-key",
            model="gpt-4o",
        )

    assert result["clip_score"] is None
