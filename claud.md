# claude.md — Core Agentic AI: Orchestration & Fact-Check Agent

## Project Overview

**Owner:** Shantam Sharma
**Role:** Core Agentic AI Engineer — Orchestration (LangGraph) + Fact-Check Agent
**Sprint:** 15-Day Agile

This project builds a multi-agent fact-checking pipeline. A claim (text + optional image) enters the system, is routed through memory and live search, cross-checked against image captions, and exits as a structured verdict JSON with explainable reasoning.

---

## Codebase Structure

```
project/
├── claude.md                        # This file
├── schemas/
│   └── contracts.json               # Shared JSON Schema — single source of truth
├── agents/
│   ├── orchestrator.py              # LangGraph graph definition (Task 2)
│   ├── fact_check_rag.py            # RAG-based fact-check agent (Task 3)
│   ├── fact_check_live.py           # Live search fact-check agent (Task 4)
│   └── cross_modal.py               # Cross-modal consistency check (Task 5)
├── memory/
│   └── memory_client.py             # Interface to Memory Agent (Vector DB + KG)
├── prompts/
│   ├── verdict_synthesis.txt        # Main verdict synthesis prompt (versioned)
│   ├── decomposition.txt            # Claim decomposition prompt
│   ├── debate_advocate.txt          # Multi-agent debate advocate prompt
│   └── cross_modal_check.txt        # Cross-modal consistency prompt
├── evaluation/
│   └── benchmark.py                 # Macro-F1, Precision@k, Entity Extraction F1 (Task 6)
├── tests/
│   └── test_contracts.py            # Schema validation tests for all handoffs
├── .env.example
├── requirements.txt
└── README.md
```

> **Note:** Directories for `explainability/`, `security/`, and `mlops/` will be added in a future sprint. Placeholder sections are marked below.

---

## Agent Interfaces (Task 1 — JSON Contracts)

All schemas live in `schemas/contracts.json` and are the **single source of truth** across all engineers.

### Input: Preprocessing → Fact-Check Agent
```json
{
  "claim_id": "string (UUID)",
  "claim_text": "string",
  "entities": ["string"],
  "source_url": "string (URL)",
  "image_caption": "string | null",
  "timestamp": "string (ISO 8601)"
}
```

### Output: Fact-Check Agent → Frontend / Memory
```json
{
  "claim_id": "string (UUID)",
  "verdict": "Supported | Refuted | Unverifiable | Conflicting",
  "confidence_score": "integer (0–100)",
  "evidence_links": ["string (URL)"],
  "reasoning": "string",
  "bias_score": "float (0–1)",
  "cross_modal_flag": "boolean",
  "cross_modal_explanation": "string | null"
}
```

### Memory Query Request / Response
```json
// Request
{ "claim_text": "string", "entities": ["string"], "top_k": "integer" }

// Response
{
  "results": [
    { "claim_id": "string", "verdict": "string", "confidence_score": "integer", "similarity": "float" }
  ],
  "cache_hit": "boolean"
}
```

---

## Task-by-Task Implementation Guide

### Task 1 — Lock JSON Interface Contracts
**Status:** Blocking — do before writing any agent code.

- [ ] Finalize schemas above with Chen Sigen and Full-Stack Engineer
- [ ] Publish `schemas/contracts.json` to shared team repo
- [ ] Write `tests/test_contracts.py` to validate every agent handoff against schemas using `jsonschema`

```python
# tests/test_contracts.py
import jsonschema, json

with open("schemas/contracts.json") as f:
    schema = json.load(f)

def validate_input(payload):
    jsonschema.validate(instance=payload, schema=schema["fact_check_input"])

def validate_output(payload):
    jsonschema.validate(instance=payload, schema=schema["fact_check_output"])
```

**Sync required with:** Chen Sigen (Preprocessing output fields) + Full-Stack Engineer (verdict format)

---

### Task 2 — LangGraph Orchestration Graph
**Status:** Blocking — defines the execution graph all other agents plug into.

- [ ] Define full graph in `agents/orchestrator.py`
- [ ] Implement router: if Memory confidence > 80 → return cached verdict, else → live search
- [ ] Define LangGraph state schema (shared state object across all nodes)
- [ ] Wire LangSmith / Langfuse from Day 1 for trace logging
- [ ] Build against mock data first; connect real agents once ready

```python
# agents/orchestrator.py
from langgraph.graph import StateGraph
from typing import TypedDict

class PipelineState(TypedDict):
    claim_id: str
    claim_text: str
    entities: list[str]
    source_url: str
    image_caption: str | None
    memory_result: dict | None
    search_results: list[dict]
    verdict: dict | None

graph = StateGraph(PipelineState)

# Node execution order:
# receive_claim → query_memory → router →
#   ├── return_cached       (confidence > 80)
#   └── live_search → cross_modal_check → synthesize_verdict → write_memory → emit_output

graph.add_node("receive_claim", receive_claim)
graph.add_node("query_memory", query_memory)
graph.add_node("router", router)
graph.add_node("return_cached", return_cached)
graph.add_node("live_search", live_search)
graph.add_node("cross_modal_check", cross_modal_check)
graph.add_node("synthesize_verdict", synthesize_verdict)
graph.add_node("write_memory", write_memory)
graph.add_node("emit_output", emit_output)
```

**Dependency:** Mock Memory responses until Chen's Memory Agent is ready (Day 5).

**HITL Note (from board):** Source credibility is tracked as a graph where edges represent credibility scores per topic:
```python
# Example credibility graph structure
source_credibility = {
    'wion': {'war': 0.8, 'hollywood': 0.4},
    'bbc':  {'war': 0.5}
}
# 'source' → 'topic' [edges = credibility score]
```

---

### Task 3 — Fact-Check Agent: RAG Path
**Status:** Dependent on Memory Agent (Day 5).

- [ ] Query Vector DB with claim embedding; retrieve top-k relevant chunks
- [ ] Feed retrieved context + claim into verdict synthesis prompt with CoT reasoning
- [ ] Enforce strict JSON output format for deterministic parsing
- [ ] Handle retrieval failure gracefully — fall through to live search if Memory returns empty or low-relevance results

```python
# agents/fact_check_rag.py
from memory.memory_client import query_memory

def fact_check_rag(state: PipelineState) -> PipelineState:
    results = query_memory(state["claim_text"], state["entities"], top_k=5)
    if not results or max(r["similarity"] for r in results) < 0.6:
        return {**state, "memory_result": None}  # fall through to live search
    # synthesize verdict from retrieved context
    ...
```

**SOTA Enhancements (implement after baseline works):**

- **GraphRAG:** Extract entities → query KG for 1–2 hop neighbors → merge graph context with vector chunks → feed unified context into verdict prompt. Expected gain: +5–10% Macro-F1 on entity-heavy claims.
- **Self-RAG:** Two prompt steps — (1) "Does this claim require external evidence?" (2) "Rate each chunk 1–5 for relevance. Exclude below 3." Expected gain: reduced hallucination + token efficiency.

---

### Task 4 — Fact-Check Agent: Live Search Path
**Status:** Triggered when RAG returns no useful prior verdict.

- [ ] Integrate Tavily API (preferred) or Serper as the search backend
- [ ] Fetch results from minimum 3 distinct sources
- [ ] Instruct LLM to weigh source credibility and flag when sources contradict each other
- [ ] Write new verdict back to Memory after synthesis

```python
# agents/fact_check_live.py
from tavily import TavilyClient

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def fact_check_live(state: PipelineState) -> PipelineState:
    results = client.search(state["claim_text"], max_results=5)
    # weigh credibility, flag contradictions, synthesize verdict
    ...
```

**SOTA Enhancements (implement after baseline works):**

- **Claim Decomposition:** Decompose compound claims into atomic sub-claims → verify each independently → aggregate with weighted confidence. Expected gain: +8–12% on LIAR mixed-label claims.
- **Multi-Agent Debate:** If 35 < confidence < 65 → spawn Advocate A (Supported) + Advocate B (Refuted) + Arbiter C. ~3x token cost, only on ambiguous claims.

**Dependency:** Write-back to Memory Agent — coordinate write API with Chen Sigen.

---

### Task 5 — Cross-Modal Consistency Check
**Status:** Dependent on objective VLM image captions from Chen Sigen.

- [ ] Single LLM call: present [CLAIM TEXT] and [IMAGE CAPTION] with clearly labelled sections
- [ ] Prompt LLM to identify only clear logical conflicts — not stylistic mismatches
- [ ] Set `cross_modal_flag` (boolean) + `cross_modal_explanation` (one line) in verdict output
- [ ] Handle null image captions gracefully — skip check and note absence in output

```python
# agents/cross_modal.py
def cross_modal_check(state: PipelineState) -> PipelineState:
    if not state.get("image_caption"):
        return {**state, "cross_modal_flag": False, "cross_modal_explanation": "No image caption provided"}
    # LLM call: compare claim vs caption, detect logical conflicts
    ...
```

**Requires from Chen:** `image_caption` must be objective, physical VLM descriptions (no subjective language).

**SOTA Enhancement (implement after baseline works):**

- **CLIP-Based Scoring:** Encode image + claim text using `openai/clip-vit-base-patch32` → compute cosine similarity → combine with LLM flag for final `cross_modal_flag`. Expected gain: higher precision out-of-context detection.

---

### Task 6 — Prompt Tuning & Benchmark Evaluation
**Status:** Begin after feature freeze (Day 11).

- [ ] Run LIAR dataset through full pipeline → compute Macro-F1, Precision@k, Entity Extraction F1
- [ ] Identify worst-performing verdict class (typically "Half-True" / "Mostly True" edge cases)
- [ ] Add few-shot examples of hard cases into `prompts/verdict_synthesis.txt`
- [ ] Add CoT prefix: `"Think step by step before concluding"`
- [ ] Version-control all prompt changes — structural changes re-trigger eval scripts

**Coordinate with Full-Stack Engineer:** Share verdict JSON format early so their eval scripts can parse it correctly.

---

## Future Workstreams

The following directories and concerns will be added in a later sprint. The codebase structure is designed to accommodate them without refactoring existing agents.

```
project/
├── explainability/         # Verdict explanation cards, confidence decomposition, CoT audit logs
├── agents/
│   └── security/           # Prompt injection defense, schema validation, adversarial source detection
└── mlops/                  # Pipeline versioning, regression testing, drift detection, cost tracking
```

---

## Key Dependencies

| Dependency | Owner | Needed By |
|---|---|---|
| Finalized JSON schema | Chen Sigen | Day 1–2 (blocking) |
| Memory Agent (Vector DB + KG) | Chen Sigen | Day 5 (blocking) |
| Objective VLM image captions | Chen Sigen | Task 5 |
| Verdict JSON | Shantam | Day 8 → Full-Stack Engineer |
| LangSmith traces | Shantam | Day 10 → Full-Stack Engineer |

---

## Environment Variables

```bash
# .env.example
ANTHROPIC_API_KEY=
TAVILY_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=core-agentic-ai
VECTOR_DB_URL=       # Chen's Memory Agent endpoint
MEMORY_WRITE_URL=    # Write-back endpoint
NEO4J_URI=           # For GraphRAG (SOTA - Task 3)
NEO4J_USERNAME=
NEO4J_PASSWORD=
```

---

## Requirements

```
langgraph
langchain
langchain-anthropic
langsmith
tavily-python
jsonschema
transformers        # CLIP cross-modal scoring (SOTA - Task 5)
torch
sentence-transformers
datasets            # LIAR benchmark
scikit-learn        # Macro-F1 computation
python-dotenv
```

---

## Sprint Milestones

| Day | Milestone |
|---|---|
| 1–2 | JSON contracts finalized and published; schema validation tests passing |
| 3–4 | LangGraph graph skeleton wired with mock data; LangSmith traces live |
| 5 | Memory Agent available — swap mock → real Memory client |
| 6–7 | RAG + Live Search paths implemented and tested |
| 8 | Verdict JSON emitted; write-back to Memory working; shared with Full-Stack Engineer |
| 9–10 | Cross-modal check implemented; LangSmith traces shared |
| 11 | Feature freeze |
| 12–13 | LIAR benchmark runs; prompt tuning based on results |
| 14–15 | Final review; all schemas, traces, and eval results documented |