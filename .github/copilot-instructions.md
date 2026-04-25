---
name: fact-check-agent-workspace
description: "Multi-agent fact-checking system with LangGraph orchestration. Use when: building agents, implementing graph nodes, adding tools, writing tests, working with memory integration, or modifying the fact-check pipeline."
---

# Fact-Checking Agent — Workspace Instructions

**Project**: Agentic fact-checking system built on LangGraph + Neo4j + ChromaDB  
**Status**: ~90% complete; architecture stable • Ready for enhancement  
**Owner**: Shantam Sharma (Core Agentic AI Engineer)

---

## Quick Start

### Build & Verify Setup
```bash
# Terminal: Setup environment (one-time)
source .venv/bin/activate
cd /home/shantam/fakenews

# Start Docker services (Neo4j localhost:7474, ChromaDB localhost:8000)
sudo ./start.sh

# Install/verify dependencies
pip install -r fact_check_agent/requirements.txt
```

### Run Tests & Validate
```bash
# All tests
PYTHONPATH=memory_agent:fact_check_agent pytest fact_check_agent/tests/ -v

# Range of specific tests
pytest fact_check_agent/tests/test_router.py::test_debate_check_routes_low_confidence -v
pytest fact_check_agent/tests/test_graph_integration.py -v  # Full flow tests
```

### Run Benchmarks
```bash
# Baseline eval (Factify2 val split, ~15 min)
./run_benchmark.sh

# Custom: LIAR train set with 500 claims
./run_benchmark.sh --dataset liar --split train --limit 500

# Output: results/benchmark_<split>_<timestamp>.csv + metrics JSON
```

---

## Codebase at a Glance

### Directory Structure
```
fact_check_agent/
├── src/
│   ├── config.py               # Standalone settings (14 flags for SOTA features)
│   ├── memory_client.py        # MemoryAgent singleton entry point
│   ├── pipeline.py             # Real pipeline: PreprocessingOutput → FactCheckOutput
│   ├── models/
│   │   ├── schemas.py          # ✅ Frozen JSON contracts (FactCheckInput/Output)
│   │   └── state.py            # 14-field LangGraph state machine
│   ├── graph/
│   │   ├── graph.py            # Build & compile graph from nodes + memory
│   │   ├── nodes.py            # 9 deterministic nodes (linear flow)
│   │   └── router.py           # One conditional edge: debate_check()
│   ├── agents/
│   │   ├── context_claim_agent.py   # Evidence gathering: Q-gen → coverage → search → summarize
│   │   └── reflection_agent.py      # Source credibility (no LLM)
│   ├── tools/
│   │   ├── rag_tool.py         # Vector search + verdict retrieval
│   │   ├── live_search_tool.py # Tavily with domain-diversity retry
│   │   ├── cross_modal_tool.py # SigLIP/Gemma/caption conflict detection
│   │   ├── freshness_tool.py   # Single-call + ReAct classification
│   │   └── reranker.py         # RRF merge + optional cross-encoder
│   ├── prompts.py              # All LLM prompt templates
│   ├── llm_factory.py          # Provider toggle: OpenAI vs Ollama
│   └── id_utils.py
├── benchmark/
│   ├── run_eval.py             # End-to-end evaluation (Macro-F1, Precision, Recall)
│   ├── generate_captions.py    # Pre-generate VLM captions (offline optimization)
│   ├── seed_hitl_graph.py      # Seed Neo4j from LIAR train set
│   └── record.py               # BenchmarkRecord data model
├── tests/
│   ├── conftest.py             # Fixture: make_memory_mock(), make_fact_check_input()
│   ├── test_router.py          # ✅ Router edge decisions (debate_check, freshness_router)
│   ├── test_data_contracts.py  # ✅ Field types, name mappings, percentage formatting
│   ├── test_graph_integration.py   # ⚠️ Needs T3, T4 cache-path tests
│   ├── test_rag_tool.py        # Vector search, confidence formatting
│   ├── test_live_search_tool.py    # Tavily retry, domain counting
│   ├── test_freshness_tool.py  # Freshness classification (LLM mocked)
│   ├── test_cross_modal_tool.py    # SigLIP/caption conflict
│   ├── test_prompts.py         # Prompt rendering, variable substitution
│   ├── test_reranker.py        # RRF merit + cross-encoder ranking
│   └── test_reflection_agent.py    # Source credibility queries
├── requirements.txt
└── .env                        # Copy from .env.example; fill in API keys

memory_agent/                   # ✅ External dependency — DO NOT MODIFY
├── src/
│   ├── models/verdict.py       # Verdict data contract (fact_check_agent.write_memory imports)
│   ├── agent.py                # MemoryAgent class
│   └── config.py               # Settings re-used by fact_check_agent
└── ...
```

---

## Architecture & Data Flow

### Claim Processing Pipeline
```
FactCheckInput (claim_id, claim_text, entities, image_caption)
           ↓
    [LangGraph]
    /nodes in order/:
      1. receive_claim       ← Entry; reset state to defaults
      2. query_memory        ← ChromaDB search + entity context
      3. freshness_check_all ← Tag results as fresh/stale
      4. context_claim_agent ← Evidence gathering loop
      5. synthesize_verdict  ← LLM: weighted by credibility
      6. [debate_check router] ← Conditional:
         ├─ low confidence → multi_agent_debate (3 LLM calls)
         └─ high confidence → skip
      7. cross_modal_check   ← SigLIP/Gemma image consistency
      8. write_memory        ← Verdict + source credibility to Neo4j/ChromaDB
      9. emit_output         ← Final FactCheckOutput
           ↓
    FactCheckOutput (verdict, confidence_score, reasoning, cross_modal_flag)
```

### State Machine (FactCheckState)
- **Input**: `input` (FactCheckInput) — set before graph.invoke()
- **Query Results**: `memory_results`, `entity_context`, `fresh_context`, `stale_context`
- **Evidence**: `context_claims` (list of {type, question, content, source})
- **SOTA Intermediate**: `sub_claims`, `debate_transcript`, `source_credibility`
- **Cross-Modal**: `cross_modal_flag`, `cross_modal_explanation`, `clip_similarity_score`
- **Output**: `output` (FactCheckOutput) — set by synthesize_verdict node

### Entity Context & Source Credibility (via Reflection Agent)
- **Entity Context**: Historical verdicts about entities in the claim (no new verdicts; read-only)
- **Source Credibility**: k-NN query over past verdicts from same source — weighted by semantic distance
  - Returns None values if < 2 samples (not "0%")
  - Used to weigh synthesis prompt context
  - Updated after every verdict write via `update_source_credibility()`

---

## Key Patterns & Conventions

### 1. Dependency Injection: MemoryAgent Singleton
**Why**: Neo4j driver connections are pooled once per process — never recreate.

```python
# ✅ Correct
from fact_check_agent.src.memory_client import get_memory
memory = get_memory()  # Singleton

# In graph nodes: closures over memory
def make_node(memory):
    def my_node(state):
        results = memory.search_similar_claims(...)
        return {...state update...}
    return my_node

# ❌ Wrong
def my_node(state):
    memory = MemoryAgent()  # Creates duplicate connection!
    ...
```

### 2. Evidence Gathering: context_claim_agent Question-Driven Loop
The agent runs **4 sequential steps per claim**:

```python
1. Q-gen: "Generate 3 factual + 3 counter-factual questions" → ["Q1", "Q2", ...]
2. Coverage: "Which questions are answered by fresh/stale memory context?" → {answered: [Q1], unanswered: [Q2,Q3]}
3. Search: For each unanswered question:
   - If context_claims already contain an answer → reuse
   - Else: Call Tavily live_search(question)
4. Summarize: Each search result → {type: "factual"|"counter"..., question, content, source}

Returns: Flat list of context_claims (all mixed together)
Input to: synthesize_verdict LLM call
```

### 3. Configuration & Feature Gating
All SOTA features are **env var flags** (default False). Disable in production if untested:

```python
# In .env:
USE_GRAPH_RAG=true              # Entity traversal + RRF merge
USE_RETRIEVAL_GATE=true         # ⚠️ Prompt exists, node NOT YET WIRED
USE_CLAIM_DECOMPOSITION=true    # ❌ Not implemented
USE_DEBATE=true                 # Multi-agent advocate/arbiter
USE_SIGLIP=true                 # SigLIP cross-modal (faster than VLM)
USE_FRESHNESS_REACT=true        # ReAct freshness loop

# Test with baseline first:
USE_GRAPH_RAG=false
USE_DEBATE=false
USE_SIGLIP=false
```

### 4. LLM Provider Strategy: Factory Pattern
**Default**: OpenAI gpt-4o

```python
# Switch providers in .env:
LLM_PROVIDER=openai             # (default)
# OR
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_LLM_MODEL=gemma4:e2b
```

**Note**: No automatic embedding re-seeding. If you switch providers mid-run, ChromaDB may have stale embeddings. Restart the Python process for clean state.

### 5. Verdict Mapping & Cross-Modal Field Naming
**Verdict Labels** (3-way classification):
- `"supported"` (from LLM "TRUE", "VERIFIED", "CORRECT")
- `"refuted"` (from LLM "FALSE", "DEBUNKED", "INCORRECT")
- `"misleading"` (fallback for ambiguous/partial responses)

**Cross-Modal Output**:
```python
# LLM returns raw confidence 0.0–1.0
output.confidence_score = int(raw_confidence * 100)  # Convert to 0–100 for JSON

# For write_memory:
memory.add_verdict(
    Verdict(
        confidence=output.confidence_score / 100,  # ✅ Back to 0.0–1.0
        image_mismatch=output.cross_modal_flag,    # Maps from cross_modal_flag
        ...
    )
)
```

### 6. Freshness Classification Logic
**Fresh** = Recent and matches current memory claim  
**Stale** = Old OR doesn't match OR missing verdict_label  

Safe default: when in doubt, return STALE (triggers live_search as fallback).

### 7. Testing: Mock MemoryAgent Pattern
```python
# In conftest.py or test file
from unittest.mock import MagicMock

def make_memory_mock(max_confidence=0.0):
    memory = MagicMock()
    memory.search_similar_claims.return_value = {
        "ids": [["clm_match001"]],
        "documents": [["Similar claim text"]],
        "metadatas": [[{
            "verdict_label": "supported",
            "verdict_confidence": max_confidence,
            "verified_at": "2026-04-20T00:00:00Z"  # or None for old
        }]],
        "distances": [[0.1]]
    }
    memory.get_entity_context.return_value = []
    return memory
```

---

## Testing: Coverage & Gaps

### ✅ Well Covered (11 test files)
- Router edge decisions (debate_check with confidence thresholds)
- Data contract validation (field types, name mappings)
- Tool-level unit tests (RAG, live search, freshness, cross-modal)
- Prompt rendering & variable substitution
- Reflection agent source credibility queries

### ⚠️ Testing Gaps (TODO Priority)
See [TODO.md#testing—missing-coverage](../TODO.md) for details:

| Test ID | Description | File | Status |
|---------|-------------|------|--------|
| T1 | freshness_router variants | test_router.py | ✅ Done |
| T2 | write_memory field mapping | test_data_contracts.py | ✅ Done |
| T3 | **Cache-hit integration** (confidence ≥0.80 + fresh) | test_graph_integration.py | ❌ Missing |
| T4 | **Stale cache path** (cache hit but revalidate=True) | test_graph_integration.py | ❌ Missing |
| T5 | **Reflection credibility write** (assert into Neo4j) | test_graph_integration.py | ⚠️ Weak coverage |

**Impact**: T3 + T4 exercise cache path; T5 validates side effects. Implement in order.

---

## Known Limitations & Next Steps

### ⚠️ Currently Broken (Low Priority)
- **Retrieval Gate**: Prompt `IS_RETRIEVAL_NEEDED_PROMPT` exists in prompts.py; gate node not wired into graph. To enable: add node, wire router edge. Flag: `USE_RETRIEVAL_GATE=true`
- **Claim Decomposition**: No implementation. Flag in config but nothing runs. Design sketch in TODO.md. High effort, unclear ROI.

### 🚀 SOTA Enhancements (Designed, Partially Implemented)
1. **GraphRAG** ✅ Fully wired; enable with `USE_GRAPH_RAG=true`
2. **Multi-Agent Debate** ✅ Fully wired; enable with `USE_DEBATE=true`
3. **SigLIP Cross-Modal** ✅ Fully wired; enable with `USE_SIGLIP=true`
4. **Freshness ReAct** ✅ Fully wired; enable with `USE_FRESHNESS_REACT=true`
5. **Reflection Agent Enhancements** (backlog):
   - Topic normalization: Strip entities/dates before embedding (low ROI)
   - Recency weighting: Time-decay on source credibility (architecture allows; not coded)

---

## Common Pitfalls

### ❌ Don't Do These
| Mistake | Why | Solution |
|---------|-----|----------|
| Instantiate MemoryAgent per request in a node | Leaks Neo4j connections | Use `get_memory()` singleton |
| Modify `memory_agent/` code | Out of scope (external dependency) | Mirror types into fact_check_agent/src/models/ if needed |
| Hardcode model names in llm_factory | Breaks when switching providers | Use settings.llm_provider + factory pattern |
| Skip freshness check for "obviously old" cached claims | Misses recent reversals/updates | Always call check_freshness() for memory hits |
| Rerank results without re-computing distances | Stale scores → wrong ranking | Update distance/metadata after rerank |
| Call graph.invoke() from inside a node | Nesting breaks state isolation | Decompose to sub-functions or step-by-step orchestration |

### ⚠️ Edge Cases
- **No memory results** → Routes to live_search automatically (max_confidence=0)
- **LLM returns non-standard verdict** → Mapped to closest label ("TRUE" → "supported", "FALSE" → "refuted")
- **All prefetched chunks + no Tavily results** → Verdict synthesized from prefetch only (context_claim_agent never calls Tavily if prefetch answers all questions)
- **Source credibility < 2 samples** → Returns None values (not "0%")
- **DB offline but dry_run off** → Writes fail silently (logged)

---

## External Dependencies & Documentation Links

### Read-Only Dependencies
- **memory_agent**: Provides MemoryAgent API for graph queries. See [MEMORY_AGENT_SUMMARY.md](../MEMORY_AGENT_SUMMARY.md)
- **Neo4j + ChromaDB**: Backends for knowledge graph + vector search. Managed by Docker.
- **LangGraph**: Graph orchestration framework. Nodes are pure functions; state machine driven.

### Key Documentation
- [CODEBASE_INSTRUCTIONS.md](../CODEBASE_INSTRUCTIONS.md) — Task-by-task implementation roadmap
- [claud.md](../claud.md) — Project owner's high-level overview
- [TODO.md](../TODO.md) — Testing gaps + SOTA enhancements (prioritized backlog)
- [DATASETS.md](../DATASETS.md) — Benchmark data sources and preprocessing
- [TESTING_AND_TUNING.md](../TESTING_AND_TUNING.md) — Eval methodology + baseline metrics

### External Resources
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Tavily API**: https://tavily.com/ (search tool)
- **Neo4j Aura**: https://neo4j.com/cloud/aura/ (KG backend)
- **ChromaDB**: https://www.trychroma.com/ (vector DB backend)

---

## Quick Reference: Build/Test Commands

```bash
# Environment setup (one-time)
source .venv/bin/activate
sudo ./start.sh                  # Docker services

# Run tests
pytest fact_check_agent/tests/ -v                                  # All
pytest fact_check_agent/tests/test_router.py -v                    # Single file
pytest fact_check_agent/tests/test_graph_integration.py::test_cache_hit_routes_to_return_cached -v  # Single test

# Run benchmark
./run_benchmark.sh                                                  # Baseline (Factify2 val, 200 claims)
./run_benchmark.sh --dataset liar --split train --limit 500       # Custom dataset
./run_benchmark.sh --dataset liar --seed-only                     # Seed memory only (no evaluation)

# LLM provider switching
export LLM_PROVIDER=ollama       # Or: openai (default)
ollama serve                     # In separate terminal if using local
```

---

## Questions? Start Here

1. **Adding a new graph node?** → Read [src/graph/nodes.py](../fact_check_agent/src/graph/nodes.py) to see 9 examples.
2. **Adding a new tool?** → See [tools/](../fact_check_agent/src/tools/) structure; follow the pattern of rag_tool.py or live_search_tool.py.
3. **Writing tests?** → Use conftest fixtures; see test_graph_integration.py for full-flow examples.
4. **Debugging a failed test?** → Check logs in [logs/](../logs/) or run with `-vv` flag.
5. **Understanding the eval workflow?** → Start with [benchmark/run_eval.py](../fact_check_agent/benchmark/run_eval.py) and DATASETS.md.
