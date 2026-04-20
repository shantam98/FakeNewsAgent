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


def build_graph(memory: "MemoryAgent"):
    """Build and compile the Fact-Check LangGraph.

    Args:
        memory: A single shared MemoryAgent instance (singleton from memory_client.py).
                Closed over by nodes that need it — never instantiated inside a node.

    Returns:
        A compiled LangGraph StateGraph.
    """
    # Bind memory and settings into nodes that need them via closures
    def _query_memory(state):       return query_memory(state, memory, settings)
    def _decompose_claim(state):    return decompose_claim(state, settings)
    def _retrieval_gate(state):     return retrieval_gate(state, settings)
    def _freshness_check(state):    return freshness_check(state, settings)
    def _live_search(state):        return live_search(state, settings)
    def _synthesize_verdict(state): return synthesize_verdict(state, settings)
    def _multi_agent_debate(state): return multi_agent_debate(state, settings)
    def _cross_modal_check(state):  return cross_modal_check(state, settings)
    def _write_memory(state):       return write_memory(state, memory)

    g = StateGraph(FactCheckState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("receive_claim",      receive_claim)
    g.add_node("decompose_claim",    _decompose_claim)   # S3
    g.add_node("query_memory",       _query_memory)
    g.add_node("retrieval_gate",     _retrieval_gate)    # S2
    g.add_node("freshness_check",    _freshness_check)
    g.add_node("return_cached",      return_cached)
    g.add_node("live_search",        _live_search)
    g.add_node("rag_retrieval",      rag_retrieval)
    g.add_node("synthesize_verdict", _synthesize_verdict)
    g.add_node("multi_agent_debate", _multi_agent_debate)  # S4
    g.add_node("cross_modal_check",  _cross_modal_check)
    g.add_node("write_memory",       _write_memory)
    g.add_node("emit_output",        emit_output)

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
