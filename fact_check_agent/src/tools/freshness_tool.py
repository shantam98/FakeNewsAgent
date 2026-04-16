"""Freshness tool — classifies whether a cached verdict needs live re-verification.

This is a tool, not an agent: it makes a single LLM call and returns a
structured classification. No loops, no follow-up actions.

Called on every cache hit (confidence >= CACHE_CONFIDENCE_THRESHOLD).
Uses an LLM to reason about claim category and time elapsed since last verification.

Tuning surface:
  - FRESHNESS_CHECK_PROMPT in prompts.py: adjust day thresholds per category,
    add few-shot examples, or change the category taxonomy
  - Replace the LLM call with a fine-tuned classifier once enough labelled data exists
"""
import json
import logging
from datetime import datetime, timezone

from openai import OpenAI

from fact_check_agent.src.prompts import FRESHNESS_CHECK_PROMPT

logger = logging.getLogger(__name__)


def check_freshness(
    claim_text: str,
    verdict_label: str,
    verdict_confidence: float,
    last_verified_at: datetime,
    api_key: str,
    model: str,
) -> dict:
    """Decide whether a cached verdict is fresh enough to use without re-verification.

    Args:
        claim_text:         The new incoming claim text.
        verdict_label:      The prior verdict (supported / refuted / misleading).
        verdict_confidence: The prior verdict confidence (0.0 – 1.0).
        last_verified_at:   When the prior verdict was written to memory.
        api_key:            OpenAI API key.
        model:              LLM model name (e.g. "gpt-4o").

    Returns:
        {
            "revalidate":     bool,   # True → run live search before synthesizing
            "reason":         str,    # one-sentence explanation
            "claim_category": str,    # inferred category (for logging / analytics)
        }
    """
    now = datetime.now(timezone.utc)
    if last_verified_at.tzinfo is None:
        last_verified_at = last_verified_at.replace(tzinfo=timezone.utc)

    time_since_verified_days = (now - last_verified_at).days

    prompt = FRESHNESS_CHECK_PROMPT.format(
        claim_text=claim_text,
        verdict_label=verdict_label,
        verdict_confidence=verdict_confidence,
        time_since_verified_days=time_since_verified_days,
    )

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        logger.info(
            "freshness_tool: claim_category=%s revalidate=%s days_old=%d reason=%s",
            result.get("claim_category", "unknown"),
            result.get("revalidate"),
            time_since_verified_days,
            result.get("reason", ""),
        )
        return {
            "revalidate":     bool(result.get("revalidate", False)),
            "reason":         result.get("reason", ""),
            "claim_category": result.get("claim_category", "unknown"),
        }
    except Exception as e:
        # On any failure, default to revalidating — safer than serving a stale verdict
        logger.warning("freshness_tool failed (%s) — defaulting to revalidate=True", e)
        return {
            "revalidate":     True,
            "reason":         f"Freshness check failed ({e}); defaulting to live search.",
            "claim_category": "unknown",
        }
