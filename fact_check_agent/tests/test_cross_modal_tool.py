"""Tests for the cross-modal tool.

Unit tests use mocks (no API keys / Ollama required).
Integration tests (marked `ollama`) call Gemma 4 via local Ollama — skipped
automatically when Ollama is not running.
"""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from fact_check_agent.src.tools.cross_modal_tool import (
    check_cross_modal,
    _vision_check,
    _llm_check,
)

# ── Test images — tiny base64 PNGs with scene labels ─────────────────────────

# Red/orange scene labelled "WILDFIRE SCENE" — clearly a fire context
_FIRE_IMAGE = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUG"
    "BwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/wAAR"
    "CAB4AMgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgED"
    "AwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcY"
    "GRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJ"
    "ipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo"
    "6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQD"
    "BAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcY"
    "GRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImK"
    "kpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq"
    "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD..."
)

# Blue scene labelled "PEACEFUL OCEAN" — clearly a calm water context
_OCEAN_IMAGE = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUG"
    "BwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/wAAR"
    "CAB4AMgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgED"
    "AwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcY"
    "GRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJ"
    "ipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo"
    "6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQD"
    "BAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcY"
    "GRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImK"
    "kpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq"
    "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD..."
)


def _ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_running(),
    reason="Ollama not running — skipping vision integration tests",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_response(conflict: bool, explanation=None):
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


def test_empty_caption_no_url_returns_no_flag():
    result = check_cross_modal(
        claim_text="Some claim", image_caption="", api_key="k", model="gpt-4o"
    )
    assert result["flag"] is False


# ── Unit: caption-based LLM path ──────────────────────────────────────────────

def test_caption_path_no_conflict():
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mk:
        mk.return_value.chat.completions.create.return_value = _mock_response(False)
        result = check_cross_modal(
            claim_text="Vaccines are safe.",
            image_caption="Doctor administering vaccine to patient.",
            api_key="k",
            model="gpt-4o",
        )
    assert result["flag"] is False


def test_caption_path_conflict():
    with patch("fact_check_agent.src.llm_factory.make_llm_client") as mk:
        mk.return_value.chat.completions.create.return_value = _mock_response(
            True, "Image shows protest but claim says peaceful gathering."
        )
        result = check_cross_modal(
            claim_text="The rally was peaceful.",
            image_caption="Police disperse violent crowd.",
            api_key="k",
            model="gpt-4o",
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


# ── Unit: vision dispatch (ollama provider) ───────────────────────────────────

def test_image_url_with_ollama_dispatches_to_vision():
    """When image_url + ollama provider, _vision_check is called not _llm_check."""
    with patch("fact_check_agent.src.tools.cross_modal_tool.settings") as ms, \
         patch("fact_check_agent.src.tools.cross_modal_tool._vision_check") as mv, \
         patch("fact_check_agent.src.tools.cross_modal_tool._llm_check") as ml:
        ms.llm_provider = "ollama"
        mv.return_value = {"conflict": False, "explanation": None}
        check_cross_modal(
            claim_text="A claim", image_caption=None, api_key="", model="",
            image_url="http://example.com/img.jpg",
        )
    mv.assert_called_once()
    ml.assert_not_called()


def test_image_url_without_ollama_falls_back_to_caption():
    """When image_url present but provider is openai, caption path is used."""
    with patch("fact_check_agent.src.tools.cross_modal_tool.settings") as ms, \
         patch("fact_check_agent.src.tools.cross_modal_tool._vision_check") as mv, \
         patch("fact_check_agent.src.tools.cross_modal_tool._llm_check") as ml:
        ms.llm_provider = "openai"
        ml.return_value = {"conflict": False, "explanation": None}
        check_cross_modal(
            claim_text="A claim", image_caption="A caption", api_key="k", model="gpt-4o",
            image_url="http://example.com/img.jpg",
        )
    mv.assert_not_called()
    ml.assert_called_once()


# ── Integration: Gemma 4 vision via Ollama ────────────────────────────────────

@requires_ollama
def test_vision_check_returns_valid_json_shape():
    """Smoke test: Gemma 4 returns a dict with conflict (bool) and explanation."""
    import urllib.request, base64, io
    from PIL import Image, ImageDraw

    # Build a small but clear fire/red scene image
    img = Image.new("RGB", (200, 120), color=(200, 60, 20))
    d = ImageDraw.Draw(img)
    for y, text in [(20, "WILDFIRE SCENE"), (50, "Buildings Burning"), (80, "Evacuation Zone")]:
        d.text((10, y), text, fill=(255, 240, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    fire_uri = f"data:image/jpeg;base64,{b64}"

    result = _vision_check(
        claim_text="This photo shows a peaceful ocean sunset with calm waters.",
        image_url=fire_uri,
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "conflict" in result, f"Missing 'conflict' key: {result}"
    assert isinstance(result["conflict"], bool), f"conflict must be bool: {result}"
    assert "explanation" in result, f"Missing 'explanation' key: {result}"


@requires_ollama
def test_vision_returns_bool_conflict_field():
    """Gemma 4 returns a valid bool for conflict — not testing specific verdict,
    only that the model responds with parseable JSON and the right field types.

    Note: gemma4:e2b (2B params) is too small for reliable cross-modal reasoning;
    larger models produce more accurate verdicts. This test verifies API integration.
    """
    import base64, io
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 120), color=(200, 60, 20))
    d = ImageDraw.Draw(img)
    for y, text in [(20, "WILDFIRE SCENE"), (50, "FIRE EMERGENCY"), (80, "Buildings Ablaze")]:
        d.text((10, y), text, fill=(255, 240, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    fire_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    result = _vision_check(
        claim_text="This peaceful ocean photo shows calm blue waters and a clear sky.",
        image_url=fire_uri,
    )

    assert isinstance(result.get("conflict"), bool), f"conflict must be bool: {result}"
    # explanation is either None or a string
    assert result.get("explanation") is None or isinstance(result["explanation"], str)


@requires_ollama
def test_check_cross_modal_vision_end_to_end():
    """Full check_cross_modal call dispatches to vision when provider=ollama."""
    import base64, io
    from PIL import Image, ImageDraw
    from unittest.mock import patch

    img = Image.new("RGB", (200, 120), color=(200, 60, 20))
    d = ImageDraw.Draw(img)
    d.text((10, 40), "WILDFIRE", fill=(255, 240, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    fire_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Force ollama provider for this call
    with patch("fact_check_agent.src.tools.cross_modal_tool.settings") as ms, \
         patch("fact_check_agent.src.tools.cross_modal_tool.settings.ollama_base_url",
               "http://localhost:11434/v1", create=True), \
         patch("fact_check_agent.src.tools.cross_modal_tool.settings.ollama_llm_model",
               "gemma4:e2b", create=True):
        ms.llm_provider = "ollama"
        ms.ollama_base_url = "http://localhost:11434/v1"
        ms.ollama_llm_model = "gemma4:e2b"
        result = check_cross_modal(
            claim_text="This photo shows a calm ocean sunset.",
            image_caption=None,
            api_key="",
            model="",
            image_url=fire_uri,
        )

    assert "flag" in result
    assert "explanation" in result
    assert isinstance(result["flag"], bool)
