# TODO

---

## Testing — Missing Coverage

Items tracked in `TESTING_AND_TUNING.md`. Complete in order — each builds on the previous.

### T1 · `freshness_router` unit tests
**File:** `fact_check_agent/tests/test_router.py`  
**What:** `freshness_router` in `src/graph/router.py` is untested. Add cases for:
- `revalidation_needed=True` → returns `"stale"`
- `revalidation_needed=False` → returns `"fresh"`
- `revalidation_needed=None` → returns `"stale"` (safe default)

### T2 · `write_memory` field mapping test
**File:** `fact_check_agent/tests/test_data_contracts.py`  
**What:** Verify the `write_memory` node constructs `Verdict` correctly:
- `confidence` stored as `output.confidence_score / 100` (float), not the raw int
- `image_mismatch` maps from `cross_modal_flag`
- `evidence_summary` = `reasoning + "\n\nSources: " + joined links`

### T3 · Cache path integration test
**File:** `fact_check_agent/tests/test_graph_integration.py`  
**What:** Full graph run where memory confidence ≥ 0.80 AND freshness returns fresh → `return_cached` node exercised, live search never called.  
**How:** Mock `memory.search_similar_claims` to return a high-confidence result with a recent `verified_at`. Mock freshness LLM to return `revalidate=False`.

### T4 · Stale cache path integration test
**File:** `fact_check_agent/tests/test_graph_integration.py`  
**What:** Cache hit (confidence ≥ 0.80) but freshness returns `revalidate=True` → live search runs, `return_cached` never called.

### T5 · Reflection agent wired into integration test
**File:** `fact_check_agent/tests/test_graph_integration.py`  
**What:** After a full graph run, assert:
- `state["source_credibility"]` is populated (even if all-None when no history)
- `memory.add_source_credibility_point` was called once with correct `source_id` and `point_id = f"sc_{verdict_id}"`

---

## Missing Components

### C1 · Entity Tracker Agent
**What:** Updates `Entity.current_credibility` in Neo4j based on verdict history for claims mentioning that entity. Currently `current_credibility` is initialised at 0.5 and never updated.  
**File:** New file `memory_agent/src/agents/entity_tracker.py` + wire into `write_memory` node.  
**Depends on:** Nothing — can be built independently.

---

## Future Enhancements

Backlog of ideas that are architected for but not yet implemented.
Each item notes the file where the change lives and what it would require.

---

## Reflection Agent

### Topic Normalisation
**File:** `fact_check_agent/src/agents/reflection_agent.py`  
**What:** Before embedding a claim as its topic vector, strip named entities, dates,
and numbers so that "The 2024 election was rigged" and "The 2016 election was rigged"
cluster to the same topic region instead of two separate neighbourhoods.  
**How:** Add a pre-processing step — either a lightweight LLM call (one sentence:
"rewrite this claim removing all named entities, dates, and specific numbers") or
spaCy NER + regex. Store the normalised text in `topic_text` metadata for debugging.  
**Why it matters:** Without normalisation, a source's history is fragmented across
surface-form variants of the same topic. The k-NN query misses relevant observations.

### Recency Weighting
**File:** `fact_check_agent/src/agents/reflection_agent.py` — `query_source_credibility()`  
**What:** Add a time-decay multiplier alongside the distance weight so recent verdicts
count more than old ones.  
**How:** Each observation already stores `created_at` (ISO 8601). Add:
```python
import math
def _recency_weight(created_at_iso: str, half_life_days: float = 90.0) -> float:
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(created_at_iso)).days
    return math.exp(-math.log(2) * age / half_life_days)

weights = [
    (1.0 / (d + 1e-6)) * _recency_weight(m["created_at"])
    for d, m in zip(distances, metadatas)
]
```
`half_life_days` is a single tunable — set lower for politically volatile sources,
higher for scientific sources.  
**Why it matters:** A source that recently reformed (or deteriorated) should be
reflected in the credibility score faster than a flat average allows.

---

## SOTA Enhancements (already gated in code)

See `TESTING_AND_TUNING.md` §6 for the full list. These are already partially
implemented behind feature flags:

| Enhancement | Enable by |
|---|---|
| GraphRAG | Add `get_entity_claims()` call in `query_memory` node |
| Self-RAG | Wire `IS_RETRIEVAL_NEEDED_PROMPT` before `live_search` node |
| Claim Decomposition | Add decomposition node before `live_search` |
| Multi-Agent Debate | Uncomment `debate_check` SOTA block in `router.py` |
| CLIP Cross-Modal | Set `ENABLE_CLIP = True` in `cross_modal_tool.py` |
| Freshness classifier upgrade | Replace LLM call in `freshness_tool.py` with a ReAct agent |

---

## Entity Tracker Agent (C1)
**What:** Updates `Entity.current_credibility` in Neo4j based on verdict history
for claims mentioning that entity. Currently `current_credibility` is initialised
at 0.5 and never written after that.  
**File:** New file `memory_agent/src/agents/entity_tracker.py` + wire into `write_memory` node.  
**Depends on:** Nothing — can be built independently of the Reflection Agent.
