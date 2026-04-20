"""Tests for the cross-modal tool.

Unit tests: mocked — no API keys, Ollama, or model downloads required.
SigLIP integration: runs locally (transformers + torch), no external services.
Ollama integration: marked `requires_ollama`, auto-skipped when Ollama is down.
"""
import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest

from fact_check_agent.src.tools.cross_modal_tool import (
    check_cross_modal,
    _siglip_check,
    _vision_check,
)


# ── Shared test image factory ─────────────────────────────────────────────────

def _make_image_uri(color=(200, 60, 20), labels=None, size=(200, 120)) -> str:
    """Return a base64 data URI for a simple PIL test image."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, color=color)
    if labels:
        d = ImageDraw.Draw(img)
        for y, text in labels:
            d.text((10, y), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_running(),
    reason="Ollama not running — skipping vision LLM integration tests",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_llm_response(conflict: bool, explanation=None):
    content = json.dumps({"conflict": conflict, "explanation": explanation})
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Unit: no image data ───────────────────────────────────────────────────────

def test_no_image_data_returns_no_flag():
    result = check_cross_modal(
        claim_text="Some claim", image_caption=None, api_key="k", model="gpt-4o"
    )
    assert result["flag"] is False
    assert result["explanation"] is None
    assert result["siglip_score"] is None


def test_empty_caption_no_url_returns_no_flag():
    result = check_cross_modal(
        claim_text="Some claim", image_caption="", api_key="k", model="gpt-4o"
    )
    assert result["flag"] is False


# ── Unit: caption-based LLM path ──────────────────────────────────────────────

def test_caption_path_no_conflict():
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mk:
        mk.return_value.chat.completions.create.return_value = _mock_llm_response(False)
        result = check_cross_modal(
            claim_text="Vaccines are safe.",
            image_caption="Doctor administering vaccine to patient.",
            api_key="k", model="gpt-4o",
        )
    assert result["flag"] is False
    assert result["siglip_score"] is None


def test_caption_path_conflict():
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mk:
        mk.return_value.chat.completions.create.return_value = _mock_llm_response(
            True, "Image shows protest but claim says peaceful gathering."
        )
        result = check_cross_modal(
            claim_text="The rally was peaceful.",
            image_caption="Police disperse violent crowd.",
            api_key="k", model="gpt-4o",
        )
    assert result["flag"] is True
    assert result["explanation"] is not None


def test_caption_path_llm_failure_degrades_gracefully():
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mk:
        mk.return_value.chat.completions.create.side_effect = Exception("API error")
        result = check_cross_modal(
            claim_text="Some claim", image_caption="Some caption", api_key="k", model="gpt-4o"
        )
    assert result["flag"] is False


# ── Unit: dispatch routing ────────────────────────────────────────────────────

def test_siglip_path_takes_priority_over_vision():
    """use_siglip=True routes to _siglip_check regardless of llm_provider."""
    with patch("fact_check_agent.src.tools.cross_modal_tool.settings") as ms, \
         patch("fact_check_agent.src.tools.cross_modal_tool._siglip_check") as msiglip, \
         patch("fact_check_agent.src.tools.cross_modal_tool._vision_check") as mvision, \
         patch("fact_check_agent.src.tools.cross_modal_tool._llm_check") as mllm:
        ms.use_siglip = True
        ms.llm_provider = "ollama"
        msiglip.return_value = {"conflict": False, "explanation": None, "siglip_score": 0.8}
        check_cross_modal(
            claim_text="A claim", image_caption=None, api_key="", model="",
            image_url="data:image/jpeg;base64,/9j/fake",
        )
    msiglip.assert_called_once()
    mvision.assert_not_called()
    mllm.assert_not_called()


def test_vision_path_when_siglip_disabled_and_ollama():
    """use_siglip=False + ollama provider → _vision_check."""
    with patch("fact_check_agent.src.tools.cross_modal_tool.settings") as ms, \
         patch("fact_check_agent.src.tools.cross_modal_tool._siglip_check") as msiglip, \
         patch("fact_check_agent.src.tools.cross_modal_tool._vision_check") as mvision, \
         patch("fact_check_agent.src.tools.cross_modal_tool._llm_check") as mllm:
        ms.use_siglip = False
        ms.llm_provider = "ollama"
        mvision.return_value = {"conflict": False, "explanation": None}
        check_cross_modal(
            claim_text="A claim", image_caption=None, api_key="", model="",
            image_url="data:image/jpeg;base64,/9j/fake",
        )
    msiglip.assert_not_called()
    mvision.assert_called_once()
    mllm.assert_not_called()


def test_llm_caption_path_when_no_image_url():
    """No image_url → _llm_check regardless of provider."""
    with patch("fact_check_agent.src.tools.cross_modal_tool.settings") as ms, \
         patch("fact_check_agent.src.tools.cross_modal_tool._siglip_check") as msiglip, \
         patch("fact_check_agent.src.tools.cross_modal_tool._vision_check") as mvision, \
         patch("fact_check_agent.src.tools.cross_modal_tool._llm_check") as mllm:
        ms.use_siglip = True
        ms.llm_provider = "ollama"
        mllm.return_value = {"conflict": False, "explanation": None}
        check_cross_modal(
            claim_text="A claim", image_caption="A caption", api_key="k", model="m",
        )
    msiglip.assert_not_called()
    mvision.assert_not_called()
    mllm.assert_called_once()


def test_siglip_score_surfaces_in_return_value():
    """siglip_score from _siglip_check is passed through to the caller."""
    with patch("fact_check_agent.src.tools.cross_modal_tool.settings") as ms, \
         patch("fact_check_agent.src.tools.cross_modal_tool._siglip_check") as msiglip:
        ms.use_siglip = True
        ms.llm_provider = "openai"
        msiglip.return_value = {"conflict": True, "explanation": "low score", "siglip_score": 0.04}
        result = check_cross_modal(
            claim_text="A claim", image_caption=None, api_key="", model="",
            image_url="data:image/jpeg;base64,/9j/fake",
        )
    assert result["flag"] is True
    assert result["siglip_score"] == pytest.approx(0.04)


# ── Integration: SigLIP local model ──────────────────────────────────────────

def test_siglip_check_returns_valid_shape():
    """_siglip_check returns a dict with conflict (bool), explanation, and siglip_score (float 0-1)."""
    uri = _make_image_uri(
        color=(20, 80, 180),
        labels=[(20, "ocean"), (55, "calm water"), (90, "blue sea")],
    )
    result = _siglip_check(
        claim_text="A photo of a calm ocean with blue water.",
        image_url=uri,
    )
    assert isinstance(result, dict)
    assert isinstance(result["conflict"], bool)
    assert result["explanation"] is None or isinstance(result["explanation"], str)
    assert isinstance(result["siglip_score"], float)
    assert 0.0 <= result["siglip_score"] <= 1.0


def test_siglip_check_score_higher_for_matching_text():
    """A visually blue image should score higher against ocean text than fire text."""
    ocean_uri = _make_image_uri(color=(20, 80, 180))
    from fact_check_agent.src.tools.cross_modal_tool import _siglip_check
    r_match    = _siglip_check("a calm blue ocean with clear water", ocean_uri)
    r_mismatch = _siglip_check("a raging wildfire with flames and smoke", ocean_uri)
    assert r_match["siglip_score"] > r_mismatch["siglip_score"], (
        f"Expected match score ({r_match['siglip_score']:.4f}) > "
        f"mismatch score ({r_mismatch['siglip_score']:.4f})"
    )


def test_siglip_model_cached_across_calls():
    """_load_siglip is called only once even for multiple checks (lru_cache)."""
    from fact_check_agent.src.tools.cross_modal_tool import _load_siglip
    uri = _make_image_uri()
    _siglip_check("first call", uri)
    cache_info_before = _load_siglip.cache_info()
    _siglip_check("second call", uri)
    cache_info_after = _load_siglip.cache_info()
    assert cache_info_after.hits > cache_info_before.hits


def test_siglip_graceful_fallback_on_bad_image():
    """Corrupt base64 image data returns conflict=False without raising."""
    result = _siglip_check(
        claim_text="some claim",
        image_url="data:image/jpeg;base64,NOTVALIDBASE64!!!",
    )
    assert result["conflict"] is False
    assert result["siglip_score"] is None


# ── Integration: Gemma 4 vision via Ollama ────────────────────────────────────

@requires_ollama
def test_vision_check_returns_valid_json_shape():
    """Gemma 4 returns a dict with conflict (bool) and explanation fields."""
    uri = _make_image_uri(
        color=(200, 60, 20),
        labels=[(20, "WILDFIRE SCENE"), (55, "Buildings Burning"), (90, "Evacuation Zone")],
    )
    result = _vision_check(
        claim_text="This photo shows a peaceful ocean sunset with calm waters.",
        image_url=uri,
    )
    assert isinstance(result, dict)
    assert isinstance(result.get("conflict"), bool)
    assert result.get("explanation") is None or isinstance(result["explanation"], str)
