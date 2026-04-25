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
    JUDGE_PROMPT,
    SKEPTIC_PROMPT,
    SUPPORTER_PROMPT,
    VERDICT_SYNTHESIS_PROMPT,
)

if TYPE_CHECKING:
    from src.memory.agent import MemoryAgent  # memory_agent — type hint only

logger = logging.getLogger(__name__)


# ── Node: receive_claim ───────────────────────────────────────────────────────

def receive_claim(state: FactCheckState) -> dict:
    """Initialise all mutable state fields to defaults."""
    inp = state["input"]
    return {
        "memory_results":          None,
        "entity_context":          [],
        "fresh_context":           [],
        "stale_context":           [],
        "context_claims":          [],
        "retrieved_chunks":        list(inp.prefetched_chunks),
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
    claims = context_claim_agent.run(
        claim_text        = state["input"].claim_text,
        fresh_context     = state.get("fresh_context", []),
        prefetched_chunks = state.get("retrieved_chunks", []),
        tavily_api_key    = settings.tavily_api_key,
    )
    return {"context_claims": claims}


# ── Node: synthesize_verdict ──────────────────────────────────────────────────

# Credibility assigned to claims by source when no stored confidence is available
_SOURCE_CREDIBILITY: dict[str, float] = {
    "tavily":     0.75,
    "prefetched": 0.70,
}
_DEFAULT_CREDIBILITY = 0.65


def _format_numbered_context_claims(context_claims: list[dict]) -> str:
    """Render context_claims as a numbered list for the LLM — no credibility scores exposed."""
    lines: list[str] = []
    for i, c in enumerate(context_claims, 1):
        if c["type"] == "memory":
            prior = f" (prior verdict: {c['verdict']})" if c.get("verdict") else ""
            lines.append(f"[{i}] MEMORY — prior verified claim{prior}")
            lines.append(f"    \"{c['content']}\"")
        else:
            tag = "FACTUAL" if c["type"] == "factual" else "COUNTER-FACTUAL"
            lines.append(f"[{i}] {tag} ({c['source']})")
            if c.get("question"):
                lines.append(f"    Q: {c['question']}")
            lines.append(f"    Evidence: {c['content']}")
        lines.append("")
    return "\n".join(lines).strip() or "No evidence available."


def _get_claim_credibility(claim: dict) -> float:
    if claim["source"] == "memory" and claim.get("confidence") is not None:
        return float(claim["confidence"])
    return _SOURCE_CREDIBILITY.get(claim["source"], _DEFAULT_CREDIBILITY)


_VALID_DEGREES = {1.0, 0.5, 0.0, -0.5, -1.0}


def _compute_verdict(
    context_claims: list[dict],
    degrees: list[float],
) -> tuple[str, int, float]:
    """Compute verdict, confidence, and evidence volume via the formula:

        V = Σ(Di × Ci) / Σ|Ci|

    Di: signed degree of support (-1.0 to 1.0) returned by the LLM.
    Ci: credibility of the source (memory confidence | tavily 0.75 | prefetched 0.70).

    Verdict thresholds:  V > 0.5 → supported | V < -0.5 → refuted | else → misleading.

    Confidence blends |V| (verdict strength) with a volume factor (Σ|Ci| / 2.5):
    a single credible source is capped around 60 %; three or more can reach 97 %.

    Returns: (verdict_label, confidence_0_100, evidence_volume=Σ|Ci|)
    """
    total_weighted    = 0.0
    total_credibility = 0.0

    for i, claim in enumerate(context_claims):
        raw_d = degrees[i] if i < len(degrees) else 0.0
        d = min(1.0, max(-1.0, float(raw_d)))   # clamp; snap to nearest valid value
        c = _get_claim_credibility(claim)
        total_weighted    += d * c
        total_credibility += abs(c)

    evidence_volume = total_credibility   # Σ|Ci| — how much evidence we have

    if total_credibility == 0:
        return "misleading", 50, 0.0

    V = total_weighted / total_credibility  # [-1.0, 1.0]

    if V > 0.5:
        verdict = "supported"
    elif V < -0.5:
        verdict = "refuted"
    else:
        verdict = "misleading"

    # Confidence = verdict strength × volume factor
    # volume_factor → 1.0 as Σ|Ci| → 2.5 (≈ 3 tavily-credibility sources)
    volume_factor = min(1.0, evidence_volume / 2.5)
    confidence    = int(min(97, max(15, abs(V) * 100 * (0.4 + 0.6 * volume_factor))))

    return verdict, confidence, evidence_volume


def synthesize_verdict(state: FactCheckState, settings) -> dict:
    """Synthesise a credibility-weighted verdict using V = Σ(Di×Ci) / Σ|Ci|.

    The LLM assigns a signed Degree of Support Di ∈ {-1,-0.5,0,0.5,1} per claim
    without seeing credibility scores. Python then computes V and derives the
    verdict label + confidence from the formula.
    """
    from fact_check_agent.src.id_utils import make_id

    inp            = state["input"]
    context_claims = state.get("context_claims") or []
    numbered_block = _format_numbered_context_claims(context_claims)

    prompt = VERDICT_SYNTHESIS_PROMPT.format(
        claim_text=inp.claim_text,
        numbered_claims=numbered_block,
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
        result = {"degrees": [], "reasoning": str(e)}

    degrees   = [float(x) for x in result.get("degrees", [])]
    reasoning = result.get("reasoning", "")
    verdict, confidence, evidence_volume = _compute_verdict(context_claims, degrees)

    logger.info(
        "synthesize_verdict: V-formula → %s (confidence=%d, evidence_volume=%.2f, degrees=%s)",
        verdict, confidence, evidence_volume,
        [round(d, 1) for d in degrees[:8]],
    )

    output = FactCheckOutput(
        verdict_id              = make_id("vrd_"),
        claim_id                = inp.claim_id,
        verdict                 = verdict,
        confidence_score        = confidence,
        evidence_links          = [],
        reasoning               = reasoning,
        cross_modal_flag        = False,
        cross_modal_explanation = None,
    )
    return {
        "output":           output,
        "neutral_degrees":  degrees,
        "neutral_reasoning": reasoning,
    }


# ── Node: multi_agent_debate (S4) ─────────────────────────────────────────────

def _format_neutral_scores_block(context_claims: list[dict], degrees: list[float]) -> str:
    """Render each evidence item with its Neutral Di for the Supporter/Skeptic/Judge."""
    lines: list[str] = []
    for i, claim in enumerate(context_claims, 1):
        d = degrees[i - 1] if i - 1 < len(degrees) else 0.0
        tag = {"memory": "MEMORY", "factual": "FACTUAL", "counter_factual": "COUNTER-FACTUAL"}.get(
            claim["type"], claim["type"].upper()
        )
        lines.append(f"[{i}] {tag} | Neutral Di = {d:+.1f}")
        if claim.get("question"):
            lines.append(f"    Q: {claim['question']}")
        lines.append(f"    \"{claim['content'][:200]}\"")
        lines.append("")
    return "\n".join(lines).strip()


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    return json.loads(raw)


def multi_agent_debate(state: FactCheckState, settings) -> dict:
    """S4: 4-role structured debate for low-confidence verdicts.

    Flow:
      Role 1 — Neutral already ran (synthesize_verdict); degrees stored in state.
      Role 2 — Supporter: proposes Di boosts where neutral was too conservative.
      Role 3 — Skeptic:   proposes Di penalties where neutral missed flaws.
               (Supporter and Skeptic run independently, only see Neutral output.)
      Role 4 — Judge:     receives all three, outputs final Di per evidence item.
    """
    inp             = state["input"]
    output          = state.get("output")
    context_claims  = state.get("context_claims") or []
    neutral_degrees = state.get("neutral_degrees") or []
    numbered_block  = _format_numbered_context_claims(context_claims)
    neutral_block   = _format_neutral_scores_block(context_claims, neutral_degrees)

    client = _llm_factory.make_llm_client()
    model  = _llm_factory.llm_model_name()

    def _call(prompt_text: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return raw

    try:
        # ── Role 2: Supporter (independent of Skeptic) ────────────────────────
        supporter_raw = _call(SUPPORTER_PROMPT.format(
            claim_text=inp.claim_text,
            numbered_claims=numbered_block,
            neutral_scores_block=neutral_block,
        ))
        supporter_result = json.loads(supporter_raw)
        supporter_adj = supporter_result.get("adjustments", [])

        # ── Role 3: Skeptic (independent of Supporter) ────────────────────────
        skeptic_raw = _call(SKEPTIC_PROMPT.format(
            claim_text=inp.claim_text,
            numbered_claims=numbered_block,
            neutral_scores_block=neutral_block,
        ))
        skeptic_result = json.loads(skeptic_raw)
        skeptic_adj = skeptic_result.get("adjustments", [])

        logger.info(
            "multi_agent_debate: supporter proposed %d adjustments, skeptic proposed %d",
            len(supporter_adj), len(skeptic_adj),
        )

        # ── Role 4: Judge ─────────────────────────────────────────────────────
        judge_raw = _call(JUDGE_PROMPT.format(
            claim_text=inp.claim_text,
            numbered_claims=numbered_block,
            neutral_scores_block=neutral_block,
            supporter_adjustments=supporter_raw,
            skeptic_adjustments=skeptic_raw,
        ))
        judge_result = json.loads(judge_raw)

        # Extract final Di per evidence item from Judge's output
        final_scores = {
            item["evidence_id"]: float(item["final_D"])
            for item in judge_result.get("final_scores", [])
        }
        stalemates = sum(
            1 for item in judge_result.get("final_scores", []) if item.get("stalemate")
        )

        # Map back to ordered list aligned with context_claims
        final_degrees = [
            final_scores.get(i + 1, neutral_degrees[i] if i < len(neutral_degrees) else 0.0)
            for i in range(len(context_claims))
        ]

        verdict, confidence, evid_volume = _compute_verdict(context_claims, final_degrees)

        # Lower confidence if many stalemates
        if stalemates > 0:
            stalemate_penalty = min(15, stalemates * 5)
            confidence = max(15, confidence - stalemate_penalty)

        transcript = (
            f"=== NEUTRAL (initial Di) ===\n{neutral_block}\n\n"
            f"=== SUPPORTER ===\n{supporter_raw}\n\n"
            f"=== SKEPTIC ===\n{skeptic_raw}\n\n"
            f"=== JUDGE ===\n{judge_raw}"
        )

        debate_summary = judge_result.get("debate_summary", "")
        reasoning = f"{debate_summary}\n\n[Debate: {len(supporter_adj)} boosts, {len(skeptic_adj)} penalties, {stalemates} stalemates]"

        updated_output = output.model_copy(update={
            "verdict":          verdict,
            "confidence_score": confidence,
            "reasoning":        reasoning,
        }) if output else output

        logger.info(
            "multi_agent_debate: verdict=%s confidence=%d (stalemates=%d, "
            "supporter_adj=%d, skeptic_adj=%d)",
            verdict, confidence, stalemates, len(supporter_adj), len(skeptic_adj),
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
        bias_score      = 0.0,
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
        bias_score      = 0.0,
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
