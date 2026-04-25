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
from fact_check_agent.src.agents import context_claim_agent
from fact_check_agent.src.tools.cross_modal_tool import check_cross_modal
from fact_check_agent.src.tools.freshness_tool import check_freshness
from fact_check_agent.src.tools.rag_tool import retrieve_similar_claims
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
    VERDICT_SYNTHESIS_PROMPT,
)

if TYPE_CHECKING:
    from src.memory.agent import MemoryAgent  # memory_agent — type hint only

logger = logging.getLogger(__name__)


# ── Node: receive_claim ───────────────────────────────────────────────────────

def receive_claim(state: FactCheckState) -> dict:
    """Initialise all mutable state fields; seed sub_claims from pre-decomposed queries."""
    inp        = state["input"]
    prefetched = list(inp.prefetched_chunks)
    sub_claims = list(inp.queries)   # pre-decomposed by preprocessing; [] in single-claim mode
    return {
        "memory_results":          None,
        "entity_context":          [],
        "fresh_context":           [],
        "stale_context":           [],
        "context_claims":          [],
        "retrieved_chunks":        prefetched,
        "sub_claims":              sub_claims,
        "debate_transcript":       None,
        "source_credibility":      None,
        "cross_modal_flag":        False,
        "cross_modal_explanation": None,
        "clip_similarity_score":   None,
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
        "source_credibility": source_cred,
    }


# ── Node: freshness_check_all ─────────────────────────────────────────────────

def freshness_check_all(state: FactCheckState, settings) -> dict:
    """Tag every retrieved SimilarClaim as fresh or stale using check_freshness().

    Claims without a verified_at timestamp default to stale (safe assumption).
    Claims without a verdict_label are also marked stale — no verdict to reuse.
    """
    results = []
    if state.get("memory_results") and state["memory_results"].results:
        results = state["memory_results"].results

    if settings.offline_mode or not results:
        return {"fresh_context": [], "stale_context": []}

    fresh: list[dict] = []
    stale: list[dict] = []

    for claim in results:
        chunk = claim.model_dump()

        if not claim.verified_at or not claim.verdict_label:
            stale.append(chunk)
            continue

        freshness = check_freshness(
            claim_text         = claim.claim_text,
            verdict_label      = claim.verdict_label,
            verdict_confidence = claim.verdict_confidence or 0.5,
            last_verified_at   = claim.verified_at,
            api_key            = settings.openai_api_key,
            model              = _llm_factory.llm_model_name(),
        )
        chunk["freshness_reason"]   = freshness["reason"]
        chunk["freshness_category"] = freshness["claim_category"]

        if freshness["revalidate"]:
            stale.append(chunk)
        else:
            fresh.append(chunk)

    logger.info("freshness_check_all: %d fresh, %d stale", len(fresh), len(stale))
    return {"fresh_context": fresh, "stale_context": stale}


# ── Node: context_claim_agent ─────────────────────────────────────────────────

def context_claim_agent_node(state: FactCheckState, settings) -> dict:
    """Generate factual/counter-factual questions, check coverage, search gaps, compose claims."""
    pre_queries = state.get("sub_claims") or []
    claims = context_claim_agent.run(
        claim_text        = state["input"].claim_text,
        fresh_context     = state.get("fresh_context", []),
        prefetched_chunks = state.get("retrieved_chunks", []),
        tavily_api_key    = settings.tavily_api_key,
        pre_queries       = pre_queries or None,
    )
    return {"context_claims": claims}


# ── Node: synthesize_verdict ──────────────────────────────────────────────────

def _format_context_claims(context_claims: list[dict]) -> str:
    """Render context_claims as a structured text block for the synthesis prompt."""
    factual = [c for c in context_claims if c["type"] == "factual"]
    counter = [c for c in context_claims if c["type"] == "counter_factual"]
    memory  = [c for c in context_claims if c["type"] == "memory"]

    lines: list[str] = []

    if factual:
        lines.append("[FACTUAL EVIDENCE] — supports claim being true")
        for c in factual:
            lines.append(f"• Q: {c['question']}")
            lines.append(f"  ↳ {c['content']} [{c['source']}]")

    if counter:
        lines.append("")
        lines.append("[COUNTER-FACTUAL EVIDENCE] — challenges claim being true")
        for c in counter:
            lines.append(f"• Q: {c['question']}")
            lines.append(f"  ↳ {c['content']} [{c['source']}]")

    if memory:
        lines.append("")
        lines.append("[MEMORY CONTEXT] — prior verified claims (fresh)")
        for c in memory:
            verdict_str = ""
            if c.get("verdict"):
                conf = f" ({c['confidence']:.0%} confidence)" if c.get("confidence") else ""
                verdict_str = f" → {c['verdict']}{conf}"
            lines.append(f"• \"{c['content']}\"{verdict_str} [memory]")

    return "\n".join(lines) if lines else "No evidence available."


def synthesize_verdict(state: FactCheckState, settings) -> dict:
    """Synthesise a credibility-weighted verdict from structured context claims."""
    from fact_check_agent.src.id_utils import make_id

    inp                  = state["input"]
    context_claims       = state.get("context_claims") or []
    context_claims_block = _format_context_claims(context_claims)

    # Build source credibility note
    cred_lines   = [f"Source: {inp.source_url}"]
    sc           = state.get("source_credibility") or {}
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
        context_claims_block=context_claims_block,
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


# ── Node: multi_agent_debate (S4) ─────────────────────────────────────────────

def multi_agent_debate(state: FactCheckState, settings) -> dict:
    """S4: Advocate/arbiter debate for low-confidence verdicts.

    Gated by settings.use_debate + debate_confidence_threshold.
    Two advocate agents argue for/against, an arbiter synthesises a final verdict.
    """
    from fact_check_agent.src.id_utils import make_id

    inp            = state["input"]
    output         = state.get("output")
    evidence_block = _format_context_claims(state.get("context_claims") or [])

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

    # Derive last_verified_at from the most recent fresh memory claim
    fresh = state.get("fresh_context") or []
    last_verified_at = None
    for claim in fresh:
        ts = claim.get("verified_at")
        if ts and (last_verified_at is None or ts > last_verified_at):
            last_verified_at = ts

    updated = current_output.model_copy(update={"last_verified_at": last_verified_at})
    return {"output": updated}
