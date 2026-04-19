"""All LangGraph node functions for the Fact-Check Agent graph.

Each node is a pure function: (state) -> dict of partial state updates.
Nodes that need MemoryAgent receive it as a second argument via closure
(see graph.py build_graph()).
"""
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import fact_check_agent.src.llm_factory as _llm_factory

from fact_check_agent.src.agents.reflection_agent import (
    query_source_credibility,
    update_source_credibility,
)
from fact_check_agent.src.tools.cross_modal_tool import check_cross_modal
from fact_check_agent.src.tools.freshness_tool import check_freshness
from fact_check_agent.src.tools.live_search_tool import format_search_context, search_live
from fact_check_agent.src.tools.rag_tool import format_rag_context, retrieve_similar_claims
from fact_check_agent.src.models.schemas import (
    FactCheckOutput,
    MemoryQueryResponse,
    SimilarClaim,
)
from fact_check_agent.src.models.state import FactCheckState
from fact_check_agent.src.prompts import VERDICT_SYNTHESIS_PROMPT

if TYPE_CHECKING:
    from src.memory.agent import MemoryAgent  # memory_agent — type hint only

logger = logging.getLogger(__name__)


# ── Node: receive_claim ───────────────────────────────────────────────────────

def receive_claim(state: FactCheckState) -> dict:
    """Initialise all mutable state fields to defaults.

    FactCheckInput is already complete when graph.invoke() is called.
    This node does not fetch or transform data — it just resets state fields
    so downstream nodes can safely read them without KeyError.
    """
    prefetched = list(state["input"].prefetched_chunks)
    return {
        "memory_results":          None,
        "entity_context":          [],
        "route":                   None,
        "revalidation_needed":     None,
        "retrieved_chunks":        prefetched,
        "sub_claims":              [],
        "debate_transcript":       None,
        "source_credibility":      None,
        "cross_modal_flag":        False,
        "cross_modal_explanation": None,
        "clip_similarity_score":   None,
        "last_verified_at":        None,
        "output":                  None,
    }


# ── Node: query_memory ────────────────────────────────────────────────────────

def query_memory(state: FactCheckState, memory: "MemoryAgent") -> dict:
    """Query MemoryAgent: similar claims (vector), entity context (graph)."""
    inp = state["input"]

    similar     = retrieve_similar_claims(inp.claim_text, memory)
    entity_ctx  = memory.get_entity_context(inp.claim_id)

    results = [
        SimilarClaim(
            claim_id           = c["claim_id"],
            claim_text         = c["claim_text"],
            verdict_label      = c.get("verdict_label"),
            verdict_confidence = c.get("verdict_confidence"),
            distance           = c["distance"],
            verified_at        = c.get("verified_at"),
        )
        for c in similar
    ]

    max_confidence = max(
        (r.verdict_confidence for r in results if r.verdict_confidence is not None),
        default=0.0,
    )

    # Surface the verified_at of the best (highest-confidence) result for freshness_check
    best = next((r for r in results if r.verdict_confidence == max_confidence and r.verified_at), None)
    last_verified_at = best.verified_at if best else None

    # Query Reflection Agent for (source, topic) credibility history
    source_cred = query_source_credibility(
        claim_text = inp.claim_text,
        source_url = inp.source_url,
        memory     = memory,
    )

    logger.info(
        "query_memory: %d similar claims, max_confidence=%.2f, %d entities, "
        "source_cred_samples=%d",
        len(results), max_confidence, len(entity_ctx),
        source_cred.get("sample_count", 0),
    )

    return {
        "memory_results":    MemoryQueryResponse(results=results, max_confidence=max_confidence),
        "entity_context":    entity_ctx,
        "last_verified_at":  last_verified_at,
        "source_credibility": source_cred,
    }


# ── Node: return_cached ───────────────────────────────────────────────────────

def return_cached(state: FactCheckState) -> dict:
    """Cache-hit path: note the cached claim for context; synthesis still runs."""
    best = next(
        (r for r in state["memory_results"].results if r.verdict_label),
        None,
    )
    chunk = (
        f"[CACHE HIT] Similar verified claim: \"{best.claim_text}\" "
        f"— Prior verdict: {best.verdict_label} "
        f"({best.verdict_confidence:.0%} confidence)"
        if best else "[CACHE HIT] Prior verdict found but no claim text available."
    )
    logger.info("Cache hit path triggered")
    return {"retrieved_chunks": [chunk], "route": "cache"}


# ── Node: freshness_check ────────────────────────────────────────────────────

def freshness_check(state: FactCheckState, settings) -> dict:
    """LLM-based classifier: should the cached verdict be re-verified via live search?

    Only runs on the cache path (confidence >= CACHE_CONFIDENCE_THRESHOLD).
    If last_verified_at is unavailable, defaults to revalidate=True (safe fallback).
    Result is stored in state so the freshness_router can route accordingly.
    """
    last_verified_at = state.get("last_verified_at")
    if not last_verified_at:
        logger.info("freshness_check: no verified_at timestamp — defaulting to revalidate")
        return {"revalidation_needed": True}

    memory_results = state["memory_results"]
    best = next(
        (r for r in memory_results.results if r.verdict_label and r.verified_at),
        None,
    )
    if not best:
        return {"revalidation_needed": True}

    result = check_freshness(
        claim_text         = state["input"].claim_text,
        verdict_label      = best.verdict_label,
        verdict_confidence = best.verdict_confidence or 0.0,
        last_verified_at   = last_verified_at,
        api_key            = settings.openai_api_key,
        model              = _llm_factory.llm_model_name(),
    )
    return {"revalidation_needed": result["revalidate"]}


# ── Node: live_search ─────────────────────────────────────────────────────────

def live_search(state: FactCheckState, settings) -> dict:
    """Live path: search Tavily for current evidence.

    Skips the Tavily call when prefetched_chunks were provided (e.g. Factify2 Option A eval).
    """
    if state.get("retrieved_chunks"):
        logger.info("live_search: skipping Tavily — using %d pre-fetched chunks", len(state["retrieved_chunks"]))
        return {"route": "live_search"}
    results          = search_live(state["input"].claim_text, api_key=settings.tavily_api_key)
    context, _links  = format_search_context(results)
    logger.info("live_search: %d results", len(results))
    return {"retrieved_chunks": [context], "route": "live_search"}


# ── Node: rag_retrieval ───────────────────────────────────────────────────────

def rag_retrieval(state: FactCheckState) -> dict:
    """Augment live search chunks with RAG context from memory results."""
    rag_context = format_rag_context(
        [r.model_dump() for r in state["memory_results"].results]
    )
    return {"retrieved_chunks": list(state["retrieved_chunks"]) + [rag_context]}


# ── Node: synthesize_verdict ──────────────────────────────────────────────────

def synthesize_verdict(state: FactCheckState, settings) -> dict:
    """Call gpt-4o to synthesise a verdict from all retrieved evidence."""
    from fact_check_agent.src.id_utils import make_id

    inp            = state["input"]
    evidence_block = "\n\n".join(state["retrieved_chunks"]) or "No evidence retrieved."

    # Build source credibility note
    cred_lines = [f"Source: {inp.source_url}"]

    # Topic-conditioned source credibility from Reflection Agent
    sc = state.get("source_credibility") or {}
    cred_mean    = sc.get("credibility_mean")
    bias_mean    = sc.get("bias_mean")
    bias_std     = sc.get("bias_std")
    sample_count = sc.get("sample_count", 0)

    if cred_mean is not None and sample_count >= 2:
        cred_lines.append(
            f"Source credibility for this topic: {cred_mean:.0%} "
            f"(based on {sample_count} past verdicts)"
        )
        cred_lines.append(
            f"Source bias for this topic: {bias_mean:.2f} ± {bias_std:.2f} "
            f"(0.0 = unbiased, 1.0 = highly biased; high std = inconsistent source)"
        )
    elif sample_count == 1:
        cred_lines.append("Source credibility: only 1 prior verdict — insufficient for reliable estimate")
    else:
        cred_lines.append("Source credibility: no prior verdicts for this source")

    if state["entity_context"]:
        cred_lines.append("Entity credibility context:")
        for e in state["entity_context"]:
            cred_lines.append(
                f"  - {e['name']} ({e['entity_type']}): "
                f"credibility {e.get('current_credibility', 0.5):.2f}, "
                f"sentiment in claim: {e.get('sentiment', 'neutral')}"
            )
    source_credibility_note = "\n".join(cred_lines)

    prompt = VERDICT_SYNTHESIS_PROMPT.format(
        claim_text=inp.claim_text,
        evidence_block=evidence_block,
        source_credibility_note=source_credibility_note,
    )

    client = _llm_factory.make_llm_client()
    try:
        response = client.chat.completions.create(
            model=_llm_factory.llm_model_name(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error("Verdict synthesis failed: %s", e)
        result = {
            "verdict": "misleading",
            "confidence_score": 0,
            "bias_score": 0.5,
            "reasoning": f"Synthesis failed: {e}",
            "evidence_links": [],
        }

    output = FactCheckOutput(
        verdict_id      = make_id("vrd_"),
        claim_id        = inp.claim_id,
        verdict         = result.get("verdict", "misleading"),
        confidence_score= int(result.get("confidence_score", 0)),
        evidence_links  = result.get("evidence_links", []),
        reasoning       = result.get("reasoning", ""),
        bias_score      = float(result.get("bias_score", 0.5)),
        cross_modal_flag= False,   # filled by cross_modal_check node
        cross_modal_explanation=None,
    )

    logger.info(
        "synthesize_verdict: %s (confidence=%d)", output.verdict, output.confidence_score
    )
    return {"output": output}


# ── Node: multi_agent_debate (stub — SOTA Task 4) ─────────────────────────────

def multi_agent_debate(state: FactCheckState, settings) -> dict:
    """SOTA: Spawn advocate/arbiter agents for ambiguous claims (35 < conf < 65).

    Baseline: this node is never reached (debate_check always returns 'skip').
    Implementation placeholder — wire up when enabling SOTA enhancements.
    """
    logger.info("multi_agent_debate called (stub — not implemented in baseline)")
    return {}


# ── Node: cross_modal_check ───────────────────────────────────────────────────

def cross_modal_check(state: FactCheckState, settings) -> dict:
    """LLM-based cross-modal consistency check (+ CLIP if ENABLE_CLIP=True)."""
    inp    = state["input"]
    result = check_cross_modal(
        claim_text    = inp.claim_text,
        image_caption = inp.image_caption,
        api_key       = settings.openai_api_key,
        model         = _llm_factory.llm_model_name(),
    )

    current_output: Optional[FactCheckOutput] = state.get("output")
    updated_output = None
    if current_output:
        updated_output = current_output.model_copy(update={
            "cross_modal_flag":        result["flag"],
            "cross_modal_explanation": result["explanation"],
        })

    return {
        "cross_modal_flag":        result["flag"],
        "cross_modal_explanation": result["explanation"],
        "clip_similarity_score":   result["clip_score"],
        "output":                  updated_output or current_output,
    }


# ── Node: write_memory ────────────────────────────────────────────────────────

def write_memory(state: FactCheckState, memory: "MemoryAgent") -> dict:
    """Write the final verdict back to MemoryAgent (ChromaDB + Neo4j)."""
    from src.models.verdict import Verdict  # memory_agent model — path set by bootstrap

    output: Optional[FactCheckOutput] = state.get("output")
    if not output:
        logger.warning("write_memory called with no output — skipping")
        return {}

    evidence_summary = output.reasoning
    if output.evidence_links:
        evidence_summary += "\n\nSources: " + " | ".join(output.evidence_links)

    verdict = Verdict(
        verdict_id      = output.verdict_id,
        claim_id        = output.claim_id,
        label           = output.verdict,
        confidence      = output.confidence_score / 100,
        evidence_summary= evidence_summary,
        bias_score      = output.bias_score,
        image_mismatch  = output.cross_modal_flag,
        verified_at     = datetime.now(timezone.utc),
    )

    memory.add_verdict(verdict)
    logger.info("write_memory: verdict %s written for claim %s", output.verdict, output.claim_id)

    # Reflection Agent: append one (source, topic, credibility, bias) observation
    update_source_credibility(
        claim_text      = state["input"].claim_text,
        source_url      = state["input"].source_url,
        verdict_id      = output.verdict_id,
        verdict_label   = output.verdict,
        confidence_score= output.confidence_score,
        bias_score      = output.bias_score,
        memory          = memory,
    )

    return {}


# ── Node: emit_output ─────────────────────────────────────────────────────────

def emit_output(state: FactCheckState) -> dict:
    """Terminal node — stamps freshness metadata onto the output before returning."""
    current_output: Optional[FactCheckOutput] = state.get("output")
    if not current_output:
        logger.error("emit_output reached with no output in state")
        return {}

    updated = current_output.model_copy(update={
        "last_verified_at":    state.get("last_verified_at"),
        "revalidation_needed": bool(state.get("revalidation_needed")),
    })
    return {"output": updated}
