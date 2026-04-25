"""Context Claim Agent — question-driven evidence gathering.

Steps (run once per claim):
  1. Generate 3 factual + 3 counter-factual questions about the claim.
  2. Check whether fresh memory context answers each question.
  3. For unanswered questions: search Tavily (live) or extract from prefetched chunks (benchmark).
  4. Summarise search results into a single relevant claim statement per question.

Returns a flat list of context_claim dicts consumed by synthesize_verdict.

Each context_claim has:
  type        — "factual" | "counter_factual" | "memory"
  question    — the question this addresses (None for memory claims)
  content     — evidence text
  verdict     — prior verdict label (memory claims only)
  confidence  — prior confidence (memory claims only)
  source      — "memory" | "tavily" | "prefetched"
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import fact_check_agent.src.llm_factory as _llm_factory
from fact_check_agent.src.prompts import (
    CONTEXT_COVERAGE_PROMPT,
    QUESTION_GENERATION_PROMPT,
    TAVILY_SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


# ── Step 1: question generation ───────────────────────────────────────────────

def _generate_questions(claim_text: str, model: str, client) -> dict[str, list[str]]:
    prompt = QUESTION_GENERATION_PROMPT.format(claim_text=claim_text)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(resp.choices[0].message.content)
        factual   = [str(q) for q in result.get("factual", [])[:3]]
        counter   = [str(q) for q in result.get("counter_factual", [])[:3]]
        logger.info("context_claim_agent: generated %d factual + %d counter-factual questions",
                    len(factual), len(counter))
        return {"factual": factual, "counter_factual": counter}
    except Exception as exc:
        logger.warning("question generation failed: %s", exc)
        return {"factual": [], "counter_factual": []}


# ── Step 2: coverage check ────────────────────────────────────────────────────

def _format_context_for_coverage(fresh_context: list[dict], prefetched_chunks: list[str]) -> str:
    parts = []
    for claim in fresh_context:
        verdict_str = f" (verdict: {claim['verdict_label']})" if claim.get("verdict_label") else ""
        parts.append(f"[MEMORY] {claim['claim_text']}{verdict_str}")
    for chunk in prefetched_chunks:
        parts.append(f"[DOCUMENT] {chunk[:800]}")
    return "\n\n".join(parts) or "No context available."


def _check_coverage(
    claim_text: str,
    questions: list[str],
    fresh_context: list[dict],
    prefetched_chunks: list[str],
    model: str,
    client,
) -> list[dict]:
    if not questions:
        return []
    questions_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    context_block   = _format_context_for_coverage(fresh_context, prefetched_chunks)
    prompt = CONTEXT_COVERAGE_PROMPT.format(
        claim_text=claim_text,
        questions_block=questions_block,
        context_block=context_block,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result   = json.loads(resp.choices[0].message.content)
        coverage = result.get("coverage", [])
        answered = sum(1 for c in coverage if c.get("answered"))
        logger.info("context_claim_agent: %d/%d questions answered by context",
                    answered, len(questions))
        return coverage
    except Exception as exc:
        logger.warning("coverage check failed: %s", exc)
        return [{"question": q, "answered": False, "evidence": None} for q in questions]


# ── Step 3+4: search + summarise ─────────────────────────────────────────────

def _summarise_search(
    question: str,
    claim_text: str,
    search_text: str,
    model: str,
    client,
) -> Optional[str]:
    prompt = TAVILY_SUMMARY_PROMPT.format(
        claim_text=claim_text,
        question=question,
        search_results=search_text[:2000],
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(resp.choices[0].message.content)
        return result.get("summary") or None
    except Exception as exc:
        logger.warning("tavily summarisation failed: %s", exc)
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    claim_text: str,
    fresh_context: list[dict],
    prefetched_chunks: list[str],
    tavily_api_key: str,
    pre_queries: list[str] | None = None,
) -> list[dict]:
    """Run the context claim agent. Returns a list of context_claim dicts.

    pre_queries: pre-decomposed sub-claims from preprocessing. When provided,
    question generation is skipped and these are used directly as factual questions.
    """
    from fact_check_agent.src.tools.live_search_tool import format_search_context, search_live

    model  = _llm_factory.llm_model_name()
    client = _llm_factory.make_llm_client()

    # Step 1 — use pre-provided queries or generate fresh ones
    if pre_queries:
        questions     = {"factual": list(pre_queries), "counter_factual": []}
        all_questions = list(pre_queries)
        logger.info("context_claim_agent: using %d pre-decomposed queries (skipping generation)", len(pre_queries))
    else:
        questions     = _generate_questions(claim_text, model, client)
        all_questions = questions["factual"] + questions["counter_factual"]

    # Step 2 — check coverage against memory + prefetched document
    coverage = _check_coverage(
        claim_text, all_questions, fresh_context, prefetched_chunks, model, client
    )
    coverage_by_q = {item["question"]: item for item in coverage}

    context_claims: list[dict] = []

    # Always include fresh memory claims
    for claim in fresh_context:
        context_claims.append({
            "type":       "memory",
            "question":   None,
            "content":    claim.get("claim_text", ""),
            "verdict":    claim.get("verdict_label"),
            "confidence": claim.get("verdict_confidence"),
            "source":     "memory",
        })

    # For each question: if unanswered → search; if answered → note is already in memory/prefetched
    for q in all_questions:
        q_type = "factual" if q in questions["factual"] else "counter_factual"
        item   = coverage_by_q.get(q, {"answered": False, "evidence": None})

        if item.get("answered") and item.get("evidence"):
            # Already answered by existing context — add as a lightweight claim
            context_claims.append({
                "type":       q_type,
                "question":   q,
                "content":    item["evidence"],
                "verdict":    None,
                "confidence": None,
                "source":     "memory" if fresh_context else "prefetched",
            })
            continue

        # Unanswered — need to search
        search_text = ""
        source_tag  = "tavily"

        if prefetched_chunks:
            # Benchmark / offline mode: extract from prefetched document text
            search_text = "\n".join(prefetched_chunks)[:3000]
            source_tag  = "prefetched"
        elif tavily_api_key:
            try:
                results     = search_live(q, api_key=tavily_api_key)
                search_text, _ = format_search_context(results)
            except Exception as exc:
                logger.warning("Tavily search failed for %r: %s", q, exc)

        if not search_text:
            continue

        summary = _summarise_search(q, claim_text, search_text, model, client)
        if summary:
            context_claims.append({
                "type":       q_type,
                "question":   q,
                "content":    summary,
                "verdict":    None,
                "confidence": None,
                "source":     source_tag,
            })

    logger.info(
        "context_claim_agent: %d context claims (%d memory, %d factual, %d counter-factual)",
        len(context_claims),
        sum(1 for c in context_claims if c["type"] == "memory"),
        sum(1 for c in context_claims if c["type"] == "factual"),
        sum(1 for c in context_claims if c["type"] == "counter_factual"),
    )
    return context_claims
