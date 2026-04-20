"""Freshness tool — classifies whether a cached verdict needs live re-verification.

Two modes selected by settings.use_freshness_react:
  - Single LLM call (default): fast, uses FRESHNESS_CHECK_PROMPT directly.
  - ReAct loop (S6): LLM can call Tavily to check if topic has evolved before deciding.

Called on every cache hit (confidence >= CACHE_CONFIDENCE_THRESHOLD).
"""
import json
import logging
from datetime import datetime, timezone

import fact_check_agent.src.llm_factory as _llm_factory
from fact_check_agent.src.prompts import FRESHNESS_CHECK_PROMPT

logger = logging.getLogger(__name__)

_REACT_SYSTEM = """\
You are deciding whether a cached fact-check verdict needs live re-verification.
You have access to a search tool to check if the topic has recently changed.
Use search only when necessary — e.g. for ongoing events or rapidly-changing topics.
After reasoning, return a JSON object with keys: revalidate (bool), reason (str), claim_category (str).
"""

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_current_info",
        "description": "Search the web for current information about a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}


def _check_freshness_single(
    claim_text: str,
    verdict_label: str,
    verdict_confidence: float,
    time_since_verified_days: int,
    model: str,
) -> dict:
    """Single LLM call freshness check (baseline path)."""
    prompt = FRESHNESS_CHECK_PROMPT.format(
        claim_text=claim_text,
        verdict_label=verdict_label,
        verdict_confidence=verdict_confidence,
        time_since_verified_days=time_since_verified_days,
    )
    client = _llm_factory.make_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def _check_freshness_react(
    claim_text: str,
    verdict_label: str,
    verdict_confidence: float,
    time_since_verified_days: int,
    model: str,
    tavily_api_key: str,
) -> dict:
    """ReAct freshness agent: can call Tavily to check if topic has evolved."""
    from fact_check_agent.src.tools.live_search_tool import search_live

    user_msg = FRESHNESS_CHECK_PROMPT.format(
        claim_text=claim_text,
        verdict_label=verdict_label,
        verdict_confidence=verdict_confidence,
        time_since_verified_days=time_since_verified_days,
    )
    messages = [
        {"role": "system", "content": _REACT_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    client = _llm_factory.make_llm_client()
    for _ in range(3):  # max 3 iterations (1 search + final answer)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[_SEARCH_TOOL],
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})
            for tc in msg.tool_calls:
                query = json.loads(tc.function.arguments).get("query", claim_text)
                try:
                    results = search_live(query, api_key=tavily_api_key)
                    snippets = "\n".join(r.get("content", "")[:300] for r in results[:3])
                except Exception as search_err:
                    snippets = f"Search unavailable: {search_err}"
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      snippets or "No results found.",
                })
        else:
            # Final answer
            raw = (msg.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            return json.loads(raw)

    raise ValueError("ReAct freshness loop did not converge")


def check_freshness(
    claim_text: str,
    verdict_label: str,
    verdict_confidence: float,
    last_verified_at: datetime,
    api_key: str,
    model: str,
) -> dict:
    """Decide whether a cached verdict is fresh enough to use without re-verification.

    Dispatches to ReAct loop when settings.use_freshness_react=True, otherwise
    uses a single LLM call.

    Returns:
        {"revalidate": bool, "reason": str, "claim_category": str}
    """
    from fact_check_agent.src.config import settings

    now = datetime.now(timezone.utc)
    if last_verified_at.tzinfo is None:
        last_verified_at = last_verified_at.replace(tzinfo=timezone.utc)
    time_since_verified_days = (now - last_verified_at).days

    try:
        if settings.use_freshness_react:
            result = _check_freshness_react(
                claim_text=claim_text,
                verdict_label=verdict_label,
                verdict_confidence=verdict_confidence,
                time_since_verified_days=time_since_verified_days,
                model=model,
                tavily_api_key=settings.tavily_api_key,
            )
        else:
            result = _check_freshness_single(
                claim_text=claim_text,
                verdict_label=verdict_label,
                verdict_confidence=verdict_confidence,
                time_since_verified_days=time_since_verified_days,
                model=model,
            )

        logger.info(
            "freshness_tool(%s): claim_category=%s revalidate=%s days_old=%d reason=%s",
            "react" if settings.use_freshness_react else "single",
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
        logger.warning("freshness_tool failed (%s) — defaulting to revalidate=True", e)
        return {
            "revalidate":     True,
            "reason":         f"Freshness check failed ({e}); defaulting to live search.",
            "claim_category": "unknown",
        }
