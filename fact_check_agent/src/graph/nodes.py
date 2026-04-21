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
from fact_check_agent.src.tools.reranker import rerank_candidates
from fact_check_agent.src.models.schemas import (
    FactCheckOutput,
    MemoryQueryResponse,
    SimilarClaim,
)
from fact_check_agent.src.models.state import FactCheckState
from fact_check_agent.src.prompts import (
    ADVOCATE_PROMPT,
    ARBITER_PROMPT,
    DECOMPOSITION_PROMPT,
    IS_RETRIEVAL_NEEDED_PROMPT,
    VERDICT_SYNTHESIS_PROMPT,
)

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
        "retrieval_gate_needed":   None,
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

def query_memory(state: FactCheckState, memory: "MemoryAgent", settings=None) -> dict:
    """Query MemoryAgent: similar claims (vector) + optional GraphRAG + reranking."""
    from fact_check_agent.src.config import settings as _settings
    if settings is None:
        settings = _settings

    if settings.offline_mode:
        logger.info("query_memory: offline_mode=True — skipping all DB reads")
        return {
            "memory_results":  MemoryQueryResponse(results=[], max_confidence=0.0),
            "entity_context":  [],
            "source_credibility": {},
            "last_verified_at": None,
        }

    inp = state["input"]

    # ── Stage 1: vector similarity search ────────────────────────────────
    vector_results = retrieve_similar_claims(inp.claim_text, memory)
    entity_ctx     = memory.get_entity_context(inp.claim_id)

    # ── Stage 2: GraphRAG — expand via entity-claim traversal ────────────
    graph_results: list[dict] = []
    if settings.use_graph_rag and vector_results:
        claim_ids  = [c["claim_id"] for c in vector_results]
        entity_ids = [e["entity_id"] for e in memory.get_entity_ids_for_claims(claim_ids)]
        if entity_ids:
            graph_results = memory.get_graph_claims_for_entities(entity_ids)
            logger.info("GraphRAG: %d entities → %d graph claims", len(entity_ids), len(graph_results))

    # ── Stage 3: RRF merge + optional cross-encoder rerank ───────────────
    reranked = rerank_candidates(
        query              = inp.claim_text,
        vector_results     = vector_results,
        graph_results      = graph_results,
        use_cross_encoder  = settings.use_cross_encoder,
        cross_encoder_model= settings.cross_encoder_model,
        top_k              = settings.reranker_top_k,
    )

    results = [
        SimilarClaim(
            claim_id           = c["claim_id"],
            claim_text         = c["claim_text"],
            verdict_label      = c.get("verdict_label"),
            verdict_confidence = c.get("verdict_confidence"),
            distance           = c.get("distance", 0.0),
            verified_at        = c.get("verified_at"),
        )
        for c in reranked
    ]

    max_confidence = max(
        (r.verdict_confidence for r in results if r.verdict_confidence is not None),
        default=0.0,
    )

    best = next((r for r in results if r.verdict_confidence == max_confidence and r.verified_at), None)
    last_verified_at = best.verified_at if best else None

    source_cred = query_source_credibility(
        claim_text = inp.claim_text,
        source_url = inp.source_url,
        memory     = memory,
    )

    logger.info(
        "query_memory: %d vector + %d graph → %d reranked, max_confidence=%.2f, "
        "graph_rag=%s, cross_encoder=%s, source_cred_samples=%d",
        len(vector_results), len(graph_results), len(results), max_confidence,
        settings.use_graph_rag, settings.use_cross_encoder,
        source_cred.get("sample_count", 0),
    )

    return {
        "memory_results":     MemoryQueryResponse(results=results, max_confidence=max_confidence),
        "entity_context":     entity_ctx,
        "last_verified_at":   last_verified_at,
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

    # Normalise verdict to the 3 valid labels — model sometimes outputs variants
    _VALID_LABELS = {"supported", "refuted", "misleading"}
    raw_verdict = str(result.get("verdict", "misleading")).lower().strip()
    if raw_verdict not in _VALID_LABELS:
        # Map common variants
        if "support" in raw_verdict:
            raw_verdict = "supported"
        elif "refut" in raw_verdict or "contradict" in raw_verdict or "false" in raw_verdict:
            raw_verdict = "refuted"
        else:
            raw_verdict = "misleading"
        logger.warning("synthesize_verdict: non-standard label normalised to '%s'", raw_verdict)

    output = FactCheckOutput(
        verdict_id      = make_id("vrd_"),
        claim_id        = inp.claim_id,
        verdict         = raw_verdict,
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


# ── Node: decompose_claim (S3) ────────────────────────────────────────────────

def decompose_claim(state: FactCheckState, settings) -> dict:
    """S3: Split compound claims into atomic sub-claims before retrieval.

    Gated by settings.use_claim_decomposition. No-op in baseline.
    When enabled, populates state['sub_claims'] for downstream synthesis.
    """
    if not settings.use_claim_decomposition:
        return {}

    inp = state["input"]
    prompt = DECOMPOSITION_PROMPT.format(claim_text=inp.claim_text)
    client = _llm_factory.make_llm_client()
    try:
        response = client.chat.completions.create(
            model=_llm_factory.llm_model_name(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        sub_claims = [
            sc["text"]
            for sc in result.get("sub_claims", [])
            if sc.get("verifiable", True)
        ]
        logger.info("decompose_claim: %d sub-claims from '%s'", len(sub_claims), inp.claim_text[:60])
        return {"sub_claims": sub_claims}
    except Exception as e:
        logger.error("decompose_claim failed: %s — proceeding with original claim", e)
        return {}


# ── Node: retrieval_gate (S2) ─────────────────────────────────────────────────

def retrieval_gate(state: FactCheckState, settings) -> dict:
    """S2: Adaptive retrieval gate — skip Tavily when memory context is sufficient.

    Gated by settings.use_retrieval_gate. When disabled, always proceeds to live search.
    Sets state['retrieval_gate_needed'] which retrieval_gate_router reads.
    """
    if not settings.use_retrieval_gate:
        return {"retrieval_gate_needed": True}

    inp = state["input"]
    memory_context = ""
    if state.get("memory_results") and state["memory_results"].results:
        from fact_check_agent.src.tools.rag_tool import format_rag_context
        memory_context = format_rag_context(
            [r.model_dump() for r in state["memory_results"].results]
        )

    prompt = IS_RETRIEVAL_NEEDED_PROMPT.format(claim_text=inp.claim_text)
    if memory_context:
        prompt += f"\n\nEXISTING CONTEXT FROM MEMORY:\n{memory_context}"

    client = _llm_factory.make_llm_client()
    try:
        response = client.chat.completions.create(
            model=_llm_factory.llm_model_name(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        needed = bool(result.get("retrieval_needed", True))
        logger.info(
            "retrieval_gate: retrieval_needed=%s reason=%s",
            needed, result.get("reason", "")
        )
        return {"retrieval_gate_needed": needed}
    except Exception as e:
        logger.error("retrieval_gate failed: %s — defaulting to retrieval_needed=True", e)
        return {"retrieval_gate_needed": True}


# ── Node: multi_agent_debate (S4) ─────────────────────────────────────────────

def multi_agent_debate(state: FactCheckState, settings) -> dict:
    """S4: Advocate/arbiter debate for low-confidence verdicts.

    Gated by settings.use_debate + debate_confidence_threshold.
    Two advocate agents argue for/against, an arbiter synthesises a final verdict.
    """
    from fact_check_agent.src.id_utils import make_id

    inp            = state["input"]
    output         = state.get("output")
    evidence_block = "\n\n".join(state["retrieved_chunks"]) or "No evidence retrieved."

    client = _llm_factory.make_llm_client()

    def _call(prompt_text: str) -> str:
        resp = client.chat.completions.create(
            model=_llm_factory.llm_model_name(),
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    try:
        for_prompt = ADVOCATE_PROMPT.format(
            position="TRUE (supported)",
            position_adj="supporting",
            claim_text=inp.claim_text,
            evidence_block=evidence_block,
        )
        against_prompt = ADVOCATE_PROMPT.format(
            position="FALSE (refuted)",
            position_adj="refuting",
            claim_text=inp.claim_text,
            evidence_block=evidence_block,
        )

        argument_for     = _call(for_prompt)
        argument_against = _call(against_prompt)

        arbiter_prompt = ARBITER_PROMPT.format(
            claim_text=inp.claim_text,
            argument_for=argument_for,
            argument_against=argument_against,
        )
        arbiter_raw = _call(arbiter_prompt)

        # Strip markdown fences if present
        if arbiter_raw.startswith("```"):
            arbiter_raw = arbiter_raw.split("```")[1]
            if arbiter_raw.startswith("json"):
                arbiter_raw = arbiter_raw[4:]
        result = json.loads(arbiter_raw)

        transcript = (
            f"=== FOR (supported) ===\n{argument_for}\n\n"
            f"=== AGAINST (refuted) ===\n{argument_against}\n\n"
            f"=== ARBITER ===\n{arbiter_raw}"
        )

        if output:
            updated_output = output.model_copy(update={
                "verdict":          result.get("verdict", output.verdict),
                "confidence_score": int(result.get("confidence_score", output.confidence_score)),
                "bias_score":       float(result.get("bias_score", output.bias_score)),
                "reasoning":        result.get("reasoning", output.reasoning),
                "evidence_links":   result.get("evidence_links", output.evidence_links),
            })
        else:
            updated_output = output

        logger.info(
            "multi_agent_debate: verdict=%s confidence=%d",
            result.get("verdict"), result.get("confidence_score"),
        )
        return {"output": updated_output, "debate_transcript": transcript}

    except Exception as e:
        logger.error("multi_agent_debate failed: %s — keeping original verdict", e)
        return {}


# ── Node: cross_modal_check ───────────────────────────────────────────────────

def cross_modal_check(state: FactCheckState, settings) -> dict:
    """Cross-modal consistency check: SigLIP / Gemma4 vision / LLM caption (priority order)."""
    inp    = state["input"]
    result = check_cross_modal(
        claim_text    = inp.claim_text,
        image_caption = inp.image_caption,
        api_key       = settings.openai_api_key,
        model         = _llm_factory.llm_model_name(),
        image_url     = getattr(inp, "image_url", None),
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
        "clip_similarity_score":   result.get("siglip_score"),
        "output":                  updated_output or current_output,
    }


# ── Node: write_memory ────────────────────────────────────────────────────────

def write_memory(state: FactCheckState, memory: "MemoryAgent") -> dict:
    """Write the final verdict back to MemoryAgent (ChromaDB + Neo4j)."""
    from fact_check_agent.src.config import settings as _settings
    if _settings.dry_run or _settings.offline_mode:
        logger.info("write_memory: %s — skipping DB write",
                    "offline_mode" if _settings.offline_mode else "dry_run")
        return {}

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
