"""Baseline cross-modal consistency check — LLM prompt only.

SOTA extension (CLIP scoring) is stubbed out and gated by a feature flag.
"""
import json
import logging
from typing import Optional

from openai import OpenAI

from fact_check_agent.src.prompts import CROSS_MODAL_PROMPT

logger = logging.getLogger(__name__)

# Set to True to enable CLIP scoring alongside the LLM check (SOTA Task 5)
ENABLE_CLIP = False
CLIP_THRESHOLD = 0.25  # scores below this indicate out-of-context image use


def check_cross_modal(
    claim_text: str,
    image_caption: Optional[str],
    api_key: str,
    model: str,
) -> dict:
    """Check for logical conflicts between claim text and image caption.

    Returns:
        {"flag": bool, "explanation": str | None, "clip_score": float | None}
    """
    if not image_caption:
        logger.debug("No image caption — skipping cross-modal check")
        return {"flag": False, "explanation": None, "clip_score": None}

    llm_result = _llm_check(claim_text, image_caption, api_key, model)

    clip_score = None
    clip_flag  = False
    if ENABLE_CLIP:
        # Import lazily so baseline runs without torch installed
        clip_score = _clip_check(claim_text, image_caption)
        clip_flag  = clip_score is not None and clip_score < CLIP_THRESHOLD

    final_flag = llm_result["conflict"] or clip_flag

    explanation = llm_result.get("explanation")
    if clip_flag and not explanation:
        explanation = f"Low visual-textual similarity (CLIP score: {clip_score:.2f})"

    return {
        "flag":        final_flag,
        "explanation": explanation,
        "clip_score":  clip_score,
    }


def _llm_check(claim_text: str, image_caption: str, api_key: str, model: str) -> dict:
    client = OpenAI(api_key=api_key)
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


def _clip_check(claim_text: str, image_caption: str) -> Optional[float]:
    """Compute CLIP cosine similarity between claim text and image caption text.

    Note: In a full implementation this would encode the actual image URL.
    For now we compare claim text against caption text in CLIP's embedding space
    as a lightweight proxy (full image encoding requires the raw image URL).
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch

        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        inputs  = processor(
            text=[claim_text, image_caption],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            similarity = (features[0] @ features[1]).item()

        return float(similarity)
    except Exception as e:
        logger.warning("CLIP check failed (torch/transformers may not be installed): %s", e)
        return None
