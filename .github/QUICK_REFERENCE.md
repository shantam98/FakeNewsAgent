---
name: quick-reference
description: "Fast lookup reference for common tasks. Use when: you need to quickly find command patterns, config settings, or code examples."
---

# Quick Reference Card

## Most Common Commands

```bash
# Setup (first time)
source .venv/bin/activate && sudo ./start.sh

# Run all tests
pytest fact_check_agent/tests/ -v

# Run single test
pytest fact_check_agent/tests/test_router.py::test_debate_check_routes_low_confidence -v

# Benchmark
./run_benchmark.sh
./run_benchmark.sh --dataset liar --split train --limit 500

# View logs
tail -f logs/benchmark_*.log
```

## Essential Files to Know

| Task | File |
|------|------|
| Main instructions | `.github/copilot-instructions.md` |
| Graph nodes | `fact_check_agent/src/graph/nodes.py` |
| Data contracts | `fact_check_agent/src/models/schemas.py` |
| LLM prompts | `fact_check_agent/src/prompts.py` |
| Project overview | `claud.md` |
| Testing gaps (T1-T5) | `TODO.md` |
| Datasets guide | `DATASETS.md` |

## Key Patterns

### Dependency Injection
```python
from fact_check_agent.src.memory_client import get_memory
memory = get_memory()  # Singleton
```

### Mock MemoryAgent
```python
from unittest.mock import MagicMock
memory = MagicMock()
memory.search_similar_claims.return_value = {"ids": [[...]], ...}
```

### LLM Factory
```python
from fact_check_agent.src.llm_factory import llm_factory
llm = llm_factory.create_client()
response = llm.call(prompt)
```

## Configuration Flags (in .env)

```
USE_GRAPH_RAG=true              # GraphRAG entity traversal
USE_DEBATE=true                 # Multi-agent debate
USE_SIGLIP=true                 # SigLIP cross-modal
USE_FRESHNESS_REACT=true        # ReAct freshness check
LLM_PROVIDER=openai             # Or: ollama
```

## State Machine (FactCheckState)

**Read These**:
- `input` — FactCheckInput (claim, entities, image)
- `memory_results` — from vector search
- `fresh_context`, `stale_context` — split by age
- `context_claims` — evidence from questions/search

**Write These**:
- `output` — FactCheckOutput (verdict + confidence)
- `cross_modal_flag`, `cross_modal_explanation`

## Verdict Labels

| Label | From LLM |
|-------|----------|
| `"supported"` | "TRUE", "VERIFIED", "CORRECT" |
| `"refuted"` | "FALSE", "DEBUNKED", "INCORRECT" |
| `"misleading"` | Fallback for ambiguous |

## Testing Gap Stubs (T3, T4, T5)

**Where**: `fact_check_agent/tests/test_graph_integration.py`

**T3**: `test_cache_hit_routes_to_return_cached()` — conf ≥ 0.80, fresh → skip live search

**T4**: `test_stale_cache_triggers_live_search()` — conf ≥ 0.80, stale → run live search

**T5**: `test_reflection_credibility_written_to_memory()` — assert `add_source_credibility_point()` called

## Node Execution Order

```
1. receive_claim → 2. query_memory → 3. freshness_check_all
→ 4. context_claim_agent → 5. synthesize_verdict
→ [6. debate_check router] → 7. cross_modal_check
→ 8. write_memory → 9. emit_output
```

## Error Handling Defaults

| Scenario | Default |
|----------|---------|
| No memory results | Route to live_search |
| LLM parse fail | Map to "misleading" |
| Freshness check fail | Mark as "stale" |
| DB offline | Only log, don't crash |
| Source credibility < 2 samples | Return None (not "0%") |

## File Paths (Quick Lookup)

```
Nodes & Orchestration:
  src/graph/graph.py              ← Graph assembly
  src/graph/nodes.py              ← All 9 nodes
  src/graph/router.py             ← Routing logic

Agents & Tools:
  src/agents/context_claim_agent.py
  src/agents/reflection_agent.py
  src/tools/rag_tool.py
  src/tools/live_search_tool.py
  src/tools/cross_modal_tool.py
  src/tools/freshness_tool.py

Models & Config:
  src/models/schemas.py           ← Frozen contracts
  src/models/state.py             ← State machine
  src/config.py                   ← Settings
  src/memory_client.py            ← MemoryAgent singleton

Tests:
  tests/conftest.py               ← Fixtures
  tests/test_graph_integration.py ← End-to-end
  tests/test_router.py            ← Router decisions
  tests/test_data_contracts.py    ← Schema validation

Benchmarks:
  benchmark/run_eval.py           ← Evaluation runner
```

## When Stuck

1. **Test fails**: `pytest -vv -s` to see prints and stack trace
2. **Graph not invoking**: Check `receive_claim` node resets state
3. **LLM response parsing**: Check `json.loads()`, may need markdown stripping
4. **Memory connection**: Check `get_memory()` singleton, not creating new instance
5. **Coverage missing**: Check `conftest.py` fixtures, mock patterns
6. **SOTA flags not working**: Set in `.env`, restart Python process

## Links to Key Docs

- [Full workspace instructions](../README.md#workspace-instructions)
- [Graph node guide](../.github/instructions/graph-orchestration.instructions.md)
- [Testing strategy](../.github/instructions/testing-strategy.instructions.md)
- [Tools & agents](../.github/instructions/agents-and-tools.instructions.md)
- [CODEBASE_INSTRUCTIONS.md](../CODEBASE_INSTRUCTIONS.md) — task-by-task roadmap
- [TODO.md](../TODO.md) — testing gaps + SOTA backlog
