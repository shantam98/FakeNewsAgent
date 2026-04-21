"""LangGraph graph assembly for the Fact-Check Agent.

Usage:
    from fact_check_agent.src.memory_client import get_memory
    from fact_check_agent.src.graph.graph import build_graph

    memory = get_memory()
    graph  = build_graph(memory)
    state  = graph.invoke({"input": fact_check_input})
    output = state["output"]
"""
import logging
import time
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from fact_check_agent.src.config import settings
from fact_check_agent.src.graph.nodes import (
    cross_modal_check,
    decompose_claim,
    emit_output,
    freshness_check,
    live_search,
    multi_agent_debate,
    query_memory,
    rag_retrieval,
    receive_claim,
    retrieval_gate,
    return_cached,
    synthesize_verdict,
    write_memory,
)
from fact_check_agent.src.graph.router import (
    debate_check,
    freshness_router,
    retrieval_gate_router,
    router,
)
from fact_check_agent.src.models.state import FactCheckState

if TYPE_CHECKING:
    from src.memory.agent import MemoryAgent

logger = logging.getLogger(__name__)
pipeline_logger = logging.getLogger("pipeline")


def _timed(name: str, fn):
    """Wrap a node function with entry/exit logging and ms-level timing."""
    def wrapper(state):
        claim_id = state.get("input", {}).claim_id if hasattr(state.get("input", {}), "claim_id") else "?"
        pipeline_logger.info("  ┌─ %-22s [%s]", name, claim_id)
        t0 = time.perf_counter()
        result = fn(state)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Extract the most useful field from the result for the log line
        note = ""
        if result:
            if "output" in result and result["output"]:
                o = result["output"]
                note = f"verdict={o.verdict} conf={o.confidence_score}"
            elif "memory_results" in result and result["memory_results"]:
                mr = result["memory_results"]
                note = f"max_conf={mr.max_confidence:.2f} n={len(mr.results)}"
            elif "route" in result and result["route"]:
                note = f"route={result['route']}"
            elif "revalidation_needed" in result:
                note = f"revalidate={result['revalidation_needed']}"
            elif "retrieval_gate_needed" in result:
                note = f"gate={'pass' if result['retrieval_gate_needed'] else 'skip'}"
            elif "cross_modal_flag" in result:
                score = result.get("clip_similarity_score")
                score_str = f" siglip={score:.3f}" if score is not None else ""
                note = f"flag={result['cross_modal_flag']}{score_str}"
            elif "sub_claims" in result and result["sub_claims"]:
                note = f"sub_claims={len(result['sub_claims'])}"
            elif "retrieved_chunks" in result:
                note = f"chunks={len(result['retrieved_chunks'])}"

        pipeline_logger.info("  └─ %-22s %5.0fms  %s", name, elapsed_ms, note)
        return result
    wrapper.__name__ = name
    return wrapper


def build_graph(memory: "MemoryAgent"):
    """Build and compile the Fact-Check LangGraph.

    Args:
        memory: A single shared MemoryAgent instance (singleton from memory_client.py).
                Closed over by nodes that need it — never instantiated inside a node.

    Returns:
        A compiled LangGraph StateGraph.
    """
    # Bind memory and settings into nodes, then wrap with timing logger
    _query_memory       = _timed("query_memory",       lambda s: query_memory(s, memory, settings))
    _decompose_claim    = _timed("decompose_claim",    lambda s: decompose_claim(s, settings))
    _retrieval_gate     = _timed("retrieval_gate",     lambda s: retrieval_gate(s, settings))
    _freshness_check    = _timed("freshness_check",    lambda s: freshness_check(s, settings))
    _live_search        = _timed("live_search",        lambda s: live_search(s, settings))
    _synthesize_verdict = _timed("synthesize_verdict", lambda s: synthesize_verdict(s, settings))
    _multi_agent_debate = _timed("multi_agent_debate", lambda s: multi_agent_debate(s, settings))
    _cross_modal_check  = _timed("cross_modal_check",  lambda s: cross_modal_check(s, settings))
    _write_memory       = _timed("write_memory",       lambda s: write_memory(s, memory))
    _receive_claim      = _timed("receive_claim",      receive_claim)
    _return_cached      = _timed("return_cached",      return_cached)
    _rag_retrieval      = _timed("rag_retrieval",      rag_retrieval)
    _emit_output        = _timed("emit_output",        emit_output)

    g = StateGraph(FactCheckState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("receive_claim",      _receive_claim)
    g.add_node("decompose_claim",    _decompose_claim)
    g.add_node("query_memory",       _query_memory)
    g.add_node("retrieval_gate",     _retrieval_gate)
    g.add_node("freshness_check",    _freshness_check)
    g.add_node("return_cached",      _return_cached)
    g.add_node("live_search",        _live_search)
    g.add_node("rag_retrieval",      _rag_retrieval)
    g.add_node("synthesize_verdict", _synthesize_verdict)
    g.add_node("multi_agent_debate", _multi_agent_debate)
    g.add_node("cross_modal_check",  _cross_modal_check)
    g.add_node("write_memory",       _write_memory)
    g.add_node("emit_output",        _emit_output)

    # ── Wire edges ────────────────────────────────────────────────────────────
    g.set_entry_point("receive_claim")
    g.add_edge("receive_claim",   "decompose_claim")    # S3: no-op when disabled
    g.add_edge("decompose_claim", "query_memory")

    # Confidence router: high-confidence cache hit → freshness_check, else → retrieval_gate
    g.add_conditional_edges("query_memory", router, {
        "cache":       "freshness_check",
        "live_search": "retrieval_gate",   # S2: gate before Tavily
    })

    # Freshness router: fresh cached verdict → return_cached, stale → re-run live search
    g.add_conditional_edges("freshness_check", freshness_router, {
        "fresh": "return_cached",
        "stale": "live_search",
    })

    # S2: Retrieval gate — skip Tavily when memory context is sufficient
    g.add_conditional_edges("retrieval_gate", retrieval_gate_router, {
        "needed": "live_search",
        "skip":   "rag_retrieval",
    })

    # Live search path: fetch live evidence → augment with RAG → synthesize
    g.add_edge("live_search",   "rag_retrieval")
    g.add_edge("rag_retrieval", "synthesize_verdict")

    # Cached path rejoins after synthesize_verdict (cache chunk already in state)
    g.add_edge("return_cached", "synthesize_verdict")

    # S4: Debate check — routes low-confidence verdicts through advocate/arbiter loop
    g.add_conditional_edges("synthesize_verdict", debate_check, {
        "skip":   "cross_modal_check",
        "debate": "multi_agent_debate",
    })
    g.add_edge("multi_agent_debate", "cross_modal_check")

    # Final path for all routes
    g.add_edge("cross_modal_check", "write_memory")
    g.add_edge("write_memory",      "emit_output")
    g.add_edge("emit_output",       END)

    compiled = g.compile()
    logger.info("Fact-Check graph compiled successfully")
    return compiled
