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

## Future Enhancements

Backlog of ideas that are architected for but not yet implemented.
Each item notes the file where the change lives and what it would require.

---

## Reflection Agent Enhancements

### R1 · Topic Normalisation
**File:** `fact_check_agent/src/agents/reflection_agent.py`  
**Status:** Not implemented — topic vectors are embedded from raw claim text.

**What:** Before embedding a claim as its topic vector, strip named entities, dates, and numbers so surface variants of the same topic cluster together.

**Example:** *"The 2024 US election was rigged by mail-in ballots"* and *"The 2016 US election was rigged by Russian interference"* are about the same topic (election integrity) but embed into different neighbourhoods because of the different years and actors. A source with a history of spreading election misinformation gets no penalty on the second claim because the k-NN query finds no similar topic observations.

**How:** Lightweight LLM call — *"Rewrite this claim removing all named entities, dates, and specific numbers"* — or spaCy NER + regex. Store normalised text in `topic_text` metadata for debugging.

---

### R2 · Recency Weighting
**File:** `fact_check_agent/src/agents/reflection_agent.py` — `query_source_credibility()`  
**Status:** Not implemented — all historical verdicts are weighted equally by distance only.

**What:** Add a time-decay multiplier alongside the distance weight so recent verdicts count more than old ones.

**Example:** *InfoWars* was flagged as low-credibility throughout 2020–2022. Suppose in 2024 it undergoes editorial reform and starts publishing accurate corrections. Without recency weighting, the 3-year backlog of refuted verdicts still drags credibility down and the system over-penalises new claims from the reformed outlet. With a 90-day half-life, the recent accurate verdicts quickly dominate.  
Conversely — a previously neutral local newspaper suddenly starts publishing fabricated stories. The system should detect the deterioration within weeks, not after years of averaging.

**How:**
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
`half_life_days` is a single tunable — lower for politically volatile sources, higher for scientific journals.

---

## SOTA Enhancements (gated in code — not yet enabled)

Each item is partially implemented behind a flag or comment. Enable in order of expected impact.

---

### S1 · GraphRAG
**File:** `fact_check_agent/src/graph/nodes.py` — `query_memory` node  
**Enable:** Set `USE_GRAPH_RAG=true` in `.env`.  
**Status:** ✅ Implemented. RRF merge + optional cross-encoder reranking in `src/tools/reranker.py`. Toggle `USE_CROSS_ENCODER=true` to activate cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, already downloaded). Both flags default to `false` — enable after baseline eval.

**Verdict revision also implemented:** Verdicts now carry `status=active/superseded`. Re-running on a known claim auto-supersedes the old verdict and creates a `SUPERSEDES` edge in Neo4j. Full audit history preserved; only active verdicts surface in queries.

**What:** After finding similar claims via vector search, also traverse the Neo4j graph to pull in all past verdicts for entities mentioned in the current claim — regardless of semantic similarity.

**Example:** A new claim says *"Pfizer's COVID vaccine causes heart failure."* The vector search finds semantically similar vaccine claims. GraphRAG additionally pulls every verdict where Pfizer or COVID vaccine appears as an entity — including a previously refuted claim phrased completely differently: *"mRNA injections permanently alter DNA."* That verdict's evidence chunks are added to the context, giving the synthesiser more signal without any live search.

**Why it matters:** Vector similarity misses paraphrases and domain shifts. Graph traversal is exact — if the entity appears, the verdict is retrieved.

---

### S2 · Adaptive Retrieval Gate
**File:** `fact_check_agent/src/graph/nodes.py` — before `live_search` node  
**Enable:** Wire `IS_RETRIEVAL_NEEDED_PROMPT` as a pre-check before calling Tavily.  
**Status:** Prompt exists in `src/prompts.py` but the gating node is not wired in.

**What:** Before spending a Tavily search credit, ask the LLM: *"Given this claim and the already-retrieved context, is additional web search needed?"* If no, skip live search entirely. This is sometimes loosely called Self-RAG in the literature, but that term properly refers to a fine-tuned model that generates inline reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`) — what we have here is simpler: a prompt-based yes/no gate, no fine-tuning required.

**Example:** Claim: *"The Eiffel Tower is in Paris."* The RAG retrieval already returned a Wikipedia chunk confirming this. The gate would short-circuit the Tavily call — saving a credit and ~1s of latency — since the existing evidence is already conclusive.  
Conversely, for a breaking-news claim about an event from last week, RAG finds nothing useful and the gate correctly allows the live search to proceed.

**Why it matters:** Reduces Tavily API cost and latency for claims that are already well-covered by the knowledge base.

---

### S3 · Claim Decomposition
**File:** `fact_check_agent/src/graph/nodes.py` — new node before `live_search`  
**Enable:** Add a decomposition node that splits compound claims before retrieval.  
**Status:** Not implemented.

**What:** Break a multi-part claim into atomic sub-claims, verify each independently, then aggregate into a final verdict.

**Example:** *"The COVID vaccine was rushed, has a 40% serious adverse event rate, and was never tested on children."* This is three separate claims bundled together. A single retrieval pass may find evidence for one part and miss the others, producing an unreliable verdict. Decomposition checks each sub-claim separately:
1. *"The COVID vaccine development timeline was unusually short"* → misleading (context needed)
2. *"COVID vaccines have a 40% serious adverse event rate"* → refuted (the real rate is <2%)
3. *"COVID vaccines were never tested on children"* → refuted (paediatric trials ran 2021)

The aggregation rule (e.g., *any refuted sub-claim → overall refuted*) is configurable.

---

### S4 · Multi-Agent Debate
**File:** `fact_check_agent/src/graph/router.py` — `debate_check`  
**Enable:** Uncomment the SOTA block in `router.py` so `debate_check` routes to `multi_agent_debate` instead of always returning `"skip"`.  
**Status:** Node is implemented, router always skips it.

**What:** For low-confidence verdicts (e.g., confidence < 70), spawn two adversarial LLM agents — one arguing the claim is true, one arguing it is false — then a judge agent resolves the debate into a final verdict.

**Example:** Claim: *"5G towers were linked to COVID-19 outbreaks in the UK."* Initial synthesis returns `misleading` at 55% confidence because the evidence is ambiguous — some early papers noted geographic correlation, later debunked. The debate agent forces both sides to cite specific evidence. The judge sees that the "true" agent can only cite retracted papers while the "false" agent cites peer-reviewed rebuttals — and returns `refuted` at 88%.

**Why it matters:** Single-pass synthesis anchors on the first evidence chunk it processes. Debate forces the model to steelman both positions before deciding.

---

### S5 · Gemma 4 Vision Cross-Modal Consistency
**File:** `fact_check_agent/src/tools/cross_modal_tool.py`  
**Enable:** Set `LLM_PROVIDER=ollama` in `.env` and pass `image_url` to `check_cross_modal`.  
**Status:** ✅ Implemented. CLIP stub removed. Gemma 4 via Ollama receives the raw image URL (base64) + claim text and returns structured JSON `{"conflict": bool, "explanation": str | null}`.

**What:** Send the raw image directly to Gemma 4 via Ollama's vision API alongside the claim text. No separate captioning step — the model sees the image and the claim in one pass and identifies logical conflicts.

**Example:** An article claims *"Protesters destroyed parliament building"* alongside an image that is actually from a football riot in 2018. Gemma 4 receives the image and claim together and flags the mismatch: *"The image shows a sports crowd, not a parliament building."* No caption intermediary needed.

**Why it matters:** Eliminates the caption → compare two-step pipeline. The vision model reasons about the image and claim in a single call. Falls back to text-caption LLM check when `image_url` is not available or provider is OpenAI.

**Known limitation:** `gemma4:e2b` (2B params) is borderline for reliable cross-modal reasoning. Larger models (gemma4:12b, gemma4:27b) produce more accurate conflict detection. Integration test verifies the API wiring, not verdict accuracy.

---

### S6 · Freshness ReAct Agent
**File:** `fact_check_agent/src/tools/freshness_tool.py`  
**Enable:** Replace the single LLM call with a ReAct loop.  
**Status:** Single LLM call with a structured prompt — no tool use.

**What:** Replace the freshness classification prompt with a ReAct agent that can call tools: search for the claim's publication date, check if the topic has evolved, and look up related recent events before deciding whether to revalidate.

**Example:** Claim: *"Russia controls the Zaporizhzhia nuclear plant."* A cached verdict from 3 months ago says `supported`. The current single LLM call sees the cached timestamp and returns `fresh` because 3 months is within the configured window. A ReAct agent would instead search *"Zaporizhzhia plant status 2024"*, find that control changed hands twice since the cached verdict, and correctly return `stale` — triggering a live search re-run.

**Why it matters:** The current LLM call only looks at the timestamp. The ReAct agent can assess whether the *topic itself* has evolved, not just whether enough time has passed.

---

## Entity Tracker Agent

**Status:** Deferred — `current_credibility` is initialised at 0.5 and intentionally left static for now. Source credibility and topic credibility already cover the common case.

**When it becomes useful:** When claims about a specific person/organisation are systematically fabricated *across multiple sources* — source credibility misses this because each source is judged independently. Example:
- Source X: *"Elon Musk said Bitcoin will hit $1M by 2025"* → refuted  
- Source Y: *"Elon Musk announced Tesla going private at $500/share"* → refuted  
- Source Z: *"Elon Musk endorsed this investment platform"* → refuted  

Even if X, Y, Z are all unknown outlets with no credibility history, entity credibility would learn that *claims putting words in Elon Musk's mouth are disproportionately fake* — adding a downward prior before evidence is retrieved for any new claim mentioning him.

**File:** New `memory_agent/src/agents/entity_tracker.py` + wire into `write_memory` node.  
**Revisit:** After first benchmark results — prioritise if the pipeline struggles on fabricated-quote claims.
