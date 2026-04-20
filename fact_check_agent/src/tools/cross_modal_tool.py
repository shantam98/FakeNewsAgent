"""Cross-modal consistency tool — checks for conflicts between claim text and image.

Two modes:
  1. Vision mode (preferred): sends the raw image URL to Gemma 4 via Ollama.
     Activated when `image_url` is provided and `llm_provider == "ollama"`.
  2. Caption mode (fallback): sends claim + text caption to any LLM.
     Used when no image URL is available or when using OpenAI.

CLIP scoring (S5 original design) is removed — Gemma 4 vision supersedes it.
"""
import json
import logging
from typing import Optional

from openai import OpenAI

import fact_check_agent.src.llm_factory as _llm_factory
from fact_check_agent.src.config import settings
from fact_check_agent.src.prompts import CROSS_MODAL_PROMPT, CROSS_MODAL_VISION_PROMPT

logger = logging.getLogger(__name__)


def check_cross_modal(
    claim_text: str,
    image_caption: Optional[str],
    api_key: str,
    model: str,
    image_url: Optional[str] = None,
) -> dict:
    """Check for logical conflicts between claim text and image/caption.

    Returns:
        {"flag": bool, "explanation": str | None}
    """
    if not image_url and not image_caption:
        logger.debug("No image data — skipping cross-modal check")
        return {"flag": False, "explanation": None}

    if image_url and settings.llm_provider == "ollama":
        result = _vision_check(claim_text, image_url)
    else:
        result = _llm_check(claim_text, image_caption or "", api_key, model)

    return {
        "flag":        result.get("conflict", False),
        "explanation": result.get("explanation"),
    }


def _vision_check(claim_text: str, image_url: str) -> dict:
    """Send image + claim to Gemma 4 via Ollama vision API."""
    client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
    prompt = CROSS_MODAL_VISION_PROMPT.format(claim_text=claim_text)
    try:
        response = client.chat.completions.create(
            model=settings.ollama_llm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if model adds them despite the prompt instruction
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        logger.error("Vision cross-modal check failed: %s", e)
        return {"conflict": False, "explanation": None}


def _llm_check(claim_text: str, image_caption: str, api_key: str, model: str) -> dict:
    client = _llm_factory.make_llm_client()
    prompt = CROSS_MODAL_PROMPT.format(
        claim_text=claim_text,
        image_caption=image_caption,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error("Cross-modal LLM check failed: %s", e)
        return {"conflict": False, "explanation": None}
