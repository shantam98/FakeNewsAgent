# Future Enhancements

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

## Reflection Agent — Entity-level Credibility
**What:** Extend the reflection agent to also update `Entity.current_credibility`
in Neo4j based on how often claims mentioning that entity are supported vs refuted.  
**File:** `memory_agent/src/memory/graph_store.py` + `nodes.py` `write_memory`  
**Depends on:** Entity Tracker Agent (not yet built).
