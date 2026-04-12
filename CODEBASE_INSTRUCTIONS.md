# Fact-Check Agent — Codebase Build Instructions

> **Role:** Core Agentic AI Engineer — LangGraph Orchestration + Fact-Check Agent
> **Sprint:** 15-Day Agile
> **Stack:** Python, LangGraph, LangSmith, OpenAI (gpt-4o), Tavily, CLIP (HuggingFace), ChromaDB, Neo4j
> **Do not modify:** `memory_agent/` — treat it as a read-only external dependency.

---

## 0. Repository Structure to Create

```
fakenews/
├── memory_agent/                     # DO NOT TOUCH — existing codebase
├── fact_check_agent/
│   ├── src/
│   │   ├── config.py                 # Shared Settings (re-export from memory_agent or extend)
│   │   ├── memory_client.py          # MemoryAgent singleton — one instance for the process
│   │   ├── pipeline.py               # Real pipeline entry point: PreprocessingOutput → FactCheckOutput
│   │   ├── models/
│   │   │   ├── schemas.py            # Frozen JSON interface contracts (Task 1)
│   │   │   └── state.py              # LangGraph state schema
│   │   ├── graph/
│   │   │   ├── nodes.py              # All LangGraph node functions
│   │   │   ├── router.py             # Routing logic
│   │   │   └── graph.py              # Graph assembly + compilation (takes MemoryAgent)
│   │   ├── agents/
│   │   │   ├── rag_agent.py          # RAG path (Task 3)
│   │   │   ├── live_search_agent.py  # Live search path (Task 4)
│   │   │   └── cross_modal_agent.py  # Cross-modal check (Task 5)
│   │   ├── prompts.py                # All LLM prompt templates
│   │   └── id_utils.py               # Re-export or copy from memory_agent
│   ├── benchmark/
│   │   ├── record.py                 # BenchmarkRecord model + adapters (see DATASETS.md §3.1)
│   │   ├── generate_captions.py      # Pre-generate VLM captions for FakeNewsNet offline
│   │   ├── seed_hitl_graph.py        # Seed Neo4j speaker-credibility edges from LIAR train set
│   │   └── run_eval.py               # End-to-end benchmark runner + metrics
│   ├── tests/
│   ├── requirements.txt
│   └── .env                          # Same env vars as memory_agent + LANGSMITH_API_KEY
├── MEMORY_AGENT_SUMMARY.md
├── DATASETS.md
└── CODEBASE_INSTRUCTIONS.md
```

---

## Task 1 — Lock JSON Interface Contracts

**Do this first. Everything else depends on it.**

Create `fact_check_agent/src/models/schemas.py` with Pydantic models defining the two frozen contracts:

### Input Contract: Preprocessing → Fact-Check Agent

```python
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class EntityRef(BaseModel):
    """A named entity attached to a claim.

    Mirrors MentionSentiment from memory_agent/src/models/claim.py — kept as a
    separate type so fact_check_agent has no import dependency on memory_agent models.
    """
    entity_id: str
    name: str
    entity_type: str  # "person", "organization", "country", "location", "event", "product"
    sentiment: str    # "positive", "negative", "neutral"


class FactCheckInput(BaseModel):
    claim_id: str                    # e.g. "clm_abc123"
    claim_text: str
    entities: list[EntityRef]        # populated from Claim.entities in real pipeline;
                                     # empty list [] in benchmark path (graph fills via Neo4j)
    source_url: str
    article_id: str                  # used to look up image_caption if not pre-fetched
    image_caption: Optional[str]     # vlm_caption string; None if article has no image
    timestamp: datetime
```

> **How `image_caption` is populated:**
> - **Real pipeline:** `pipeline.py` calls `memory.get_caption_by_article(claim.article_id)` before `graph.invoke()` and sets it on `FactCheckInput` — never inside a graph node.
> - **Benchmark path:** Set on `BenchmarkRecord.image_caption` during the pre-generation step (DATASETS.md §4 Step 1), then passed through by `to_fact_check_input()`.

### Output Contract: Fact-Check Agent → Frontend / Memory

```python
class FactCheckOutput(BaseModel):
    verdict_id: str
    claim_id: str
    verdict: str                # "supported" | "refuted" | "misleading"
    confidence_score: int       # 0–100 (convert from float 0.0–1.0 at boundary)
    evidence_links: list[str]   # source URLs used as evidence
    reasoning: str              # chain-of-thought explanation
    bias_score: float           # 0.0–1.0
    cross_modal_flag: bool
    cross_modal_explanation: Optional[str]  # one-line; None if no image or no conflict
```

> **Note:** When writing to `MemoryAgent.add_verdict()`, construct a `Verdict` object (from `memory_agent/src/models/verdict.py`) with `confidence = confidence_score / 100` and `image_mismatch = cross_modal_flag`.

### Memory Query Request/Response

```python
class MemoryQueryRequest(BaseModel):
    claim_text: str
    top_k: int = 5

class MemoryQueryResponse(BaseModel):
    results: list[SimilarClaim]
    max_confidence: float       # highest confidence among returned verdicts; 0 if none

class SimilarClaim(BaseModel):
    claim_id: str
    claim_text: str
    verdict_label: Optional[str]
    verdict_confidence: Optional[float]
    distance: float             # chromadb cosine distance; lower = more similar
```

**Freeze these schemas as the single source of truth. Version them with comments if changed.**

---

## Task 2 — LangGraph Orchestration Graph

### LangGraph State Schema (`src/models/state.py`)

```python
from typing import Optional, TypedDict

from fact_check_agent.src.models.schemas import (
    FactCheckInput,
    FactCheckOutput,
    MemoryQueryResponse,
)


class FactCheckState(TypedDict):
    # ── Input (required — set before graph.invoke()) ──────────────────────
    input: FactCheckInput

    # ── Memory query results ──────────────────────────────────────────────
    memory_results: Optional[MemoryQueryResponse]    # None until query_memory runs
    entity_context: list[dict]                       # from MemoryAgent.get_entity_context()

    # ── Routing decision ──────────────────────────────────────────────────
    route: Optional[str]          # "cache" | "live_search"

    # ── Fact-check intermediate results ───────────────────────────────────
    retrieved_chunks: list[str]   # from RAG or live search
    sub_claims: list[str]         # from claim decomposition (SOTA)
    debate_transcript: Optional[str]  # from multi-agent debate (SOTA)

    # ── Cross-modal ────────────────────────────────────────────────────────
    cross_modal_flag: bool
    cross_modal_explanation: Optional[str]
    clip_similarity_score: Optional[float]

    # ── Output (set by emit_output node) ──────────────────────────────────
    output: Optional[FactCheckOutput]


# Default initial state — passed alongside FactCheckInput in graph.invoke()
INITIAL_STATE: dict = {
    "memory_results":        None,
    "entity_context":        [],
    "route":                 None,
    "retrieved_chunks":      [],
    "sub_claims":            [],
    "debate_transcript":     None,
    "cross_modal_flag":      False,
    "cross_modal_explanation": None,
    "clip_similarity_score": None,
    "output":                None,
}
```

### Execution Graph

```
receive_claim
    │
    ▼
query_memory  ←── calls MemoryAgent.search_similar_claims() + get_entity_context()
    │
    ▼
[router]
    ├── confidence > 0.80 ──→ return_cached  ──────────────────────────┐
    │                                                                    │
    └── confidence ≤ 0.80 ──→ live_search (Task 4)                     │
                                    │                                    │
                                    ▼                                    │
                              rag_retrieval (Task 3)                    │
                                    │                                    │
                                    ▼                                    │
                          synthesize_verdict                             │
                          (with chain-of-thought)                        │
                                    │                                    │
                                    ▼                                    │
                        [debate_check: 35 < conf < 65?]                 │
                          │                     │                        │
                        Yes                    No                       │
                          ▼                     │                        │
                    multi_agent_debate          │                        │
                          │                     │                        │
                          └──────────┬──────────┘                        │
                                     ▼                                   │
                             cross_modal_check (Task 5)  ◄──────────────┘
                                     │
                                     ▼
                              write_memory  ←── calls MemoryAgent.add_verdict()
                                     │
                                     ▼
                               emit_output
```

### `receive_claim` Node — Implementation (`src/graph/nodes.py`)

This is the first node in the graph. Its job is to initialise all mutable state fields to their defaults so downstream nodes can safely read them without `KeyError`.

```python
def receive_claim(state: FactCheckState) -> dict:
    """Initialise all state fields to defaults.

    FactCheckInput is already complete when graph.invoke() is called — both
    image_caption and entities are set by the caller (pipeline.py or benchmark
    run_eval.py) before invocation. This node does not fetch anything.
    """
    return {
        "memory_results":          None,
        "entity_context":          [],
        "route":                   None,
        "retrieved_chunks":        [],
        "sub_claims":              [],
        "debate_transcript":       None,
        "cross_modal_flag":        False,
        "cross_modal_explanation": None,
        "clip_similarity_score":   None,
        "output":                  None,
    }
```

### Graph Assembly + MemoryAgent Singleton (`src/graph/graph.py`)

`MemoryAgent` holds a live Neo4j driver connection — it must be created **once per process**, not per request. Use a factory function that closes over a single shared instance:

```python
# src/graph/graph.py
from langgraph.graph import StateGraph, END
from src.models.state import FactCheckState
from src.graph.nodes import (
    receive_claim, query_memory, router,
    return_cached, rag_retrieval, live_search,
    synthesize_verdict, debate_check, multi_agent_debate,
    cross_modal_check, write_memory, emit_output,
)

def build_graph(memory):
    """Build and compile the LangGraph graph.

    Args:
        memory: A single shared MemoryAgent instance. Closed over by all nodes
                that need it — never instantiated inside a node.
    Returns:
        A compiled LangGraph StateGraph ready for graph.invoke().
    """
    # Wrap nodes that need memory access as closures
    def _query_memory(state):   return query_memory(state, memory)
    def _write_memory(state):   return write_memory(state, memory)

    g = StateGraph(FactCheckState)
    g.add_node("receive_claim",    receive_claim)
    g.add_node("query_memory",     _query_memory)
    g.add_node("return_cached",    return_cached)
    g.add_node("rag_retrieval",    rag_retrieval)
    g.add_node("live_search",      live_search)
    g.add_node("synthesize_verdict", synthesize_verdict)
    g.add_node("multi_agent_debate", multi_agent_debate)
    g.add_node("cross_modal_check",  cross_modal_check)
    g.add_node("write_memory",     _write_memory)
    g.add_node("emit_output",      emit_output)

    g.set_entry_point("receive_claim")
    g.add_edge("receive_claim", "query_memory")
    g.add_conditional_edges("query_memory", router, {
        "cache":       "return_cached",
        "live_search": "live_search",
    })
    g.add_edge("live_search",       "rag_retrieval")
    g.add_edge("rag_retrieval",     "synthesize_verdict")
    g.add_conditional_edges("synthesize_verdict", debate_check, {
        "debate":      "multi_agent_debate",
        "skip":        "cross_modal_check",
    })
    g.add_edge("multi_agent_debate",  "cross_modal_check")
    g.add_edge("return_cached",       "cross_modal_check")
    g.add_edge("cross_modal_check",   "write_memory")
    g.add_edge("write_memory",        "emit_output")
    g.add_edge("emit_output",         END)

    return g.compile()
```

```python
# src/memory_client.py  — MemoryAgent singleton for the fact_check_agent process
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../memory_agent"))

from src.memory.agent import MemoryAgent
from src.config import settings

_memory: MemoryAgent | None = None

def get_memory() -> MemoryAgent:
    """Return the process-level MemoryAgent singleton. Thread-safe for read queries."""
    global _memory
    if _memory is None:
        _memory = MemoryAgent(settings)
    return _memory
```

### Implementation Notes

- Wire `LangSmith` tracing from Day 1: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=...`
- Build all nodes against **mock Memory responses first** — do not require a live MemoryAgent connection during development. Pass a mock object to `build_graph()` that returns fixture data.
- Every node function: `(state: FactCheckState) -> dict` (partial state update — return only changed keys)
- The router function reads `state["memory_results"].max_confidence` — threshold `0.80`

---

## Task 3 — Fact-Check Agent: RAG Path

File: `fact_check_agent/src/agents/rag_agent.py`

### Baseline RAG

1. Call `MemoryAgent.search_similar_claims(claim_text, top_k=5)`
2. Parse ChromaDB result dict — extract `documents[0]` (list of claim texts) and `distances[0]`
3. For each returned claim_id, call `MemoryAgent.get_verdict_by_claim(claim_id)` to get prior verdicts
4. Inject retrieved context + claim into verdict synthesis prompt with CoT prefix

### SOTA Enhancement 1: GraphRAG

After step 2, additionally:
1. Call `MemoryAgent.get_entity_context(claim_id)` — returns `[{entity_id, name, entity_type, current_credibility, sentiment}]`
2. For each entity with `current_credibility < 0.4` or `> 0.8`, call `MemoryAgent.get_entity_claims(entity_id)` to get their verdict history (1-hop traversal)
3. Merge graph context with vector chunks into a unified context block before the verdict prompt

**Context block format:**
```
[VECTOR EVIDENCE]
- Claim: "..." | Prior verdict: supported (0.92 confidence)
- Claim: "..." | Prior verdict: refuted (0.78 confidence)

[GRAPH CONTEXT]
- Entity: "Elon Musk" (person) | Credibility: 0.62 | Recent verdicts: refuted x3, supported x1
- Entity: "Twitter" (organization) | Credibility: 0.71
```

### SOTA Enhancement 2: Self-RAG

Add two prompt steps before verdict synthesis:
1. `IS_RETRIEVAL_NEEDED_PROMPT`: "Does this claim require external evidence or is it self-evidently true/false? Answer yes/no with reasoning."
2. `CHUNK_RELEVANCE_PROMPT`: "Rate each retrieved chunk 1–5 for relevance to the claim. Exclude chunks rated below 3."

Both are simple LLM calls with structured JSON output. Skip retrieval if step 1 returns `no`.

---

## Task 4 — Fact-Check Agent: Live Search Path

File: `fact_check_agent/src/agents/live_search_agent.py`

### Baseline Live Search

1. Call Tavily API: `TavilyClient(api_key).search(query=claim_text, max_results=5, search_depth="advanced")`
   - Tavily is already used in `memory_agent/src/scraper/fetchers/newsapi.py` — reference that for the API key setting
2. Require minimum 3 distinct source domains — retry with broader query if fewer returned
3. Pass results into verdict synthesis prompt with source credibility weighting instruction
4. After synthesis, call `MemoryAgent.add_verdict(verdict)` to write back

### SOTA Enhancement 3: Claim Decomposition

Add a decomposition step before search:
1. `DECOMPOSITION_PROMPT`: "List all atomic, independently falsifiable sub-claims in: '{claim_text}'. Return JSON: `{sub_claims: [{text, verifiable: bool}]}`"
2. Run search + synthesis independently for each sub-claim where `verifiable=true`
3. Aggregate sub-verdicts into a final verdict using weighted confidence:
   - All supported → supported
   - Any refuted → misleading or refuted depending on proportion
   - Average confidence scores

### SOTA Enhancement 4: Multi-Agent Debate

Triggered when `35 < confidence_score < 65` after initial synthesis:
1. **Advocate A** — `ADVOCATE_PROMPT`: "Argue that the following claim is SUPPORTED. Present the strongest evidence."
2. **Advocate B** — `ADVOCATE_PROMPT`: "Argue that the following claim is REFUTED. Present the strongest counter-evidence."
3. **Arbiter** — `ARBITER_PROMPT`: "Given these opposing arguments, produce a final ruling with revised confidence."

Each is one LLM call (gpt-4o). Pass debate transcript into `state["debate_transcript"]` for LangSmith tracing.

---

## Task 5 — Cross-Modal Consistency Check

File: `fact_check_agent/src/agents/cross_modal_agent.py`

### Baseline LLM Check

Input: `state["input"].image_caption` and `state["input"].claim_text`

```python
CROSS_MODAL_PROMPT = """
[CLAIM TEXT]
{claim_text}

[IMAGE CAPTION]
{image_caption}

Identify only clear logical conflicts between what the claim states and what the image shows.
Do not flag stylistic mismatches, tone differences, or speculative connections.

If there is a clear conflict, explain it in one sentence.
Return JSON: {"conflict": true|false, "explanation": "one sentence or null"}
"""
```

- If `image_caption` is None: set `cross_modal_flag=False`, `cross_modal_explanation=None`, skip check
- Set `state["cross_modal_flag"]` and `state["cross_modal_explanation"]` from LLM response

### SOTA Enhancement 5: CLIP Scoring

```python
from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_clip_similarity(image_url: str, claim_text: str) -> float:
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(text=[claim_text], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return outputs.logits_per_image.item()  # cosine similarity score
```

- Store result in `state["clip_similarity_score"]`
- Tune threshold on Factify dataset — start at `0.25` (below = likely out-of-context)
- Final `cross_modal_flag = llm_conflict OR (clip_score < threshold)`
- `cross_modal_explanation` uses the LLM explanation when available, otherwise `f"Low visual-textual similarity (CLIP score: {score:.2f})"`

---

## Task 6 — Prompt Tuning & Benchmark Evaluation

### Verdict Synthesis Prompt (base template)

```python
VERDICT_SYNTHESIS_PROMPT = """\
Think step by step before concluding.

CLAIM: {claim_text}

EVIDENCE:
{evidence_block}

SOURCE CREDIBILITY CONTEXT:
{source_credibility_note}

Based on the evidence, determine:
1. Is the claim SUPPORTED, REFUTED, or MISLEADING?
2. What is your confidence (0–100)?
3. What is the potential bias score (0.0–1.0)?
4. Summarise your reasoning in 2–3 sentences.

Return JSON:
{{
  "verdict": "supported|refuted|misleading",
  "confidence_score": 0-100,
  "bias_score": 0.0-1.0,
  "reasoning": "...",
  "evidence_links": ["url1", "url2"]
}}
"""
```

### Evaluation Plan

- Dataset: LIAR benchmark
- Metrics: Macro-F1, Precision@k, Entity Extraction F1
- Freeze features after Day 11, run evals, tune prompts for worst-performing class ("Half-True"/"Mostly True")
- Version-control prompt changes — structural changes re-trigger eval scripts
- Share `FactCheckOutput` JSON schema with Full-Stack Engineer for eval script compatibility

---

## Dependencies (`requirements.txt`)

```
# LangGraph / LangChain
langgraph>=0.2
langchain>=0.3
langchain-openai>=0.2
langsmith>=0.1

# LLM / Embeddings
openai>=1.40

# Search
tavily-python>=0.3

# Cross-modal CLIP
transformers>=4.40
torch>=2.2
Pillow>=10.0

# Data / Config
pydantic>=2.7
pydantic-settings>=2.3

# Memory Agent (internal) — install as editable or add to PYTHONPATH
# sys.path.insert(0, "../memory_agent")
```

---

## Environment Variables (`.env`)

```
# Inherited from memory_agent (same values)
OPENAI_API_KEY=...
NEO4J_URI=...
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
TAVILY_API_KEY=...
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# New for Fact-Check Agent
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=fakenews-factcheck
```

---

## Real Pipeline Entry Point (`src/pipeline.py`)

This is the bridge between the existing `memory_agent` pipeline and the fact-check graph. It converts each `Claim` from a `PreprocessingOutput` into a `FactCheckInput`, pre-fetches the image caption from MemoryAgent (so the graph never has to), then invokes the graph.

```python
# src/pipeline.py
from datetime import datetime, timezone
from typing import Optional

from src.memory_client import get_memory
from src.graph.graph import build_graph
from src.models.schemas import EntityRef, FactCheckInput, FactCheckOutput

# Import memory_agent types (memory_agent/ must be on PYTHONPATH)
from src.models.pipeline import PreprocessingOutput


def claim_to_fact_check_input(
    output: PreprocessingOutput,
    claim_index: int,
    image_caption: Optional[str],
) -> FactCheckInput:
    """Convert one Claim from a PreprocessingOutput into a FactCheckInput.

    Args:
        output:        The PreprocessingOutput produced by PreprocessingAgent.
        claim_index:   Which claim in output.claims to convert.
        image_caption: Pre-fetched VLM caption string, or None.
    """
    claim = output.claims[claim_index]
    return FactCheckInput(
        claim_id      = claim.claim_id,
        claim_text    = claim.claim_text,
        entities      = [
            EntityRef(
                entity_id   = e.entity_id,
                name        = e.name,
                entity_type = e.entity_type,
                sentiment   = e.sentiment,
            )
            for e in claim.entities
        ],
        source_url    = output.article.url,
        article_id    = claim.article_id,
        image_caption = image_caption,
        timestamp     = claim.extracted_at,
    )


def run_fact_check(output: PreprocessingOutput) -> list[FactCheckOutput]:
    """Run the fact-check graph on every claim in a PreprocessingOutput.

    Called after MemoryAgent.ingest_preprocessed() in the real pipeline.
    Returns one FactCheckOutput per claim.
    """
    memory = get_memory()           # singleton — not re-created per call
    graph  = build_graph(memory)    # graph can also be a module-level singleton

    # Pre-fetch image caption once for the article — shared across all claims
    caption_result = memory.get_caption_by_article(output.article.article_id)
    image_caption: Optional[str] = (
        caption_result["documents"][0]
        if caption_result.get("documents") else None
    )

    results: list[FactCheckOutput] = []
    for i in range(len(output.claims)):
        fact_check_input = claim_to_fact_check_input(output, i, image_caption)
        state = graph.invoke({"input": fact_check_input})
        results.append(state["output"])

    return results
```

---

## MemoryAgent Usage Inside Graph Nodes

The singleton is accessed via `get_memory()` in `memory_client.py` but is **passed into nodes as a closure** via `build_graph(memory)` — never called directly inside node functions. Reference pattern:

```python
# src/graph/nodes.py

def query_memory(state: FactCheckState, memory) -> dict:
    """Query MemoryAgent for similar claims and entity context."""
    inp = state["input"]
    raw = memory.search_similar_claims(inp.claim_text, top_k=5)
    entity_ctx = memory.get_entity_context(inp.claim_id)
    source_cred = memory.get_source_credibility(inp.article_id)

    # Parse ChromaDB result → MemoryQueryResponse
    results = _parse_chroma_results(raw, memory)
    return {
        "memory_results": results,
        "entity_context": entity_ctx,
    }


def write_memory(state: FactCheckState, memory) -> dict:
    """Write the final verdict back to MemoryAgent (both ChromaDB and Neo4j)."""
    from src.id_utils import make_id
    from src.models.verdict import Verdict  # memory_agent model

    output: FactCheckOutput = state["output"]

    # evidence_summary combines reasoning text + source URLs
    evidence_summary = output.reasoning
    if output.evidence_links:
        evidence_summary += "\n\nSources: " + " | ".join(output.evidence_links)

    verdict = Verdict(
        verdict_id      = make_id("vrd_"),
        claim_id        = state["input"].claim_id,
        label           = output.verdict,
        confidence      = output.confidence_score / 100,   # int 0–100 → float 0.0–1.0
        evidence_summary= evidence_summary,
        bias_score      = output.bias_score,
        image_mismatch  = output.cross_modal_flag,
        verified_at     = datetime.now(timezone.utc),
    )
    memory.add_verdict(verdict)
    return {}   # no state changes — write_memory is a side-effect node
```

---

## HITL Note (from Notion)

The Notion board includes a requirement for Human-in-the-Loop credibility scoring:

```python
# Source-topic credibility graph (seeded in Neo4j)
# Edges: Source → Topic with weight = credibility_score
# Example:
# 'wion' → 'war': 0.8
# 'bbc'  → 'war': 0.5
```

This should be incorporated as an override layer in the `router` node: before returning a cached verdict, check if the source-topic credibility warrants a forced re-verification.

---

## Development Sequence

| Day | Task |
|---|---|
| 1–2 | Task 1: Freeze schemas. Set up repo, LangSmith, mock MemoryAgent stub |
| 3–4 | Task 2: Build full LangGraph graph skeleton against mocks |
| 5–7 | Task 3: RAG path (baseline + GraphRAG + Self-RAG) |
| 7–9 | Task 4: Live search (baseline + Claim Decomposition + Multi-Agent Debate) |
| 8 | Emit first `FactCheckOutput` JSON to Full-Stack Engineer |
| 9–10 | Task 5: Cross-modal check (LLM + CLIP) |
| 10 | Connect real MemoryAgent (Chen's), end-to-end test |
| 10 | Share LangSmith/Langfuse execution traces |
| 11 | Freeze features |
| 11–14 | Task 6: Benchmark, prompt tuning |
| 15 | Final review, write-back verified |
