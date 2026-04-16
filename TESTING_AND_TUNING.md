# Testing & Tuning Guide — Fact-Check Agent

> How to test each component in isolation, run end-to-end evaluation, and tune the pipeline to improve Macro-F1.

---

## Naming conventions

| Term | Meaning in this codebase |
|---|---|
| **Tool** | A single function — one API call or one LLM call, no loops, no planning. Lives in `src/tools/`. |
| **Agent** | A multi-step, stateful orchestrator that routes between tools or accumulates knowledge across calls. Lives in `src/agents/` or `memory_agent/src/memory/`. |
| **Node** | One step inside the LangGraph pipeline. Calls one or more tools/agents and updates state. Lives in `src/graph/nodes.py`. |

---

## 0. Setup

```bash
cd fakenews
pip install -r fact_check_agent/requirements.txt
cp .env.example .env          # fill in API keys
```

Run all tests (from the repo root):
```bash
PYTHONPATH="" .venv/bin/python -m pytest fact_check_agent/tests/ -v
```

---

## 1. Unit Testing Individual Tools

Each tool in `src/tools/` is a pure function — testable without a live LLM or database.

### 1.1 RAG Tool (`src/tools/rag_tool.py`)

**What it does:** Queries ChromaDB for semantically similar past claims, fetches their verdicts, and formats them as a prompt context block. No LLM call.

**Test file:** `tests/test_rag_tool.py` ✅

```python
from fact_check_agent.src.tools.rag_tool import format_rag_context, retrieve_similar_claims

def test_format_rag_context_with_verdict():
    claims = [{
        "claim_id": "clm_1", "claim_text": "Vaccines cause autism",
        "verdict_label": "refuted", "verdict_confidence": 0.95, "distance": 0.12,
    }]
    result = format_rag_context(claims)
    assert "refuted" in result
    assert "95%" in result
```

**Toggle to tune:** Change `top_k` in `retrieve_similar_claims()` to control how many prior claims are retrieved.

---

### 1.2 Live Search Tool (`src/tools/live_search_tool.py`)

**What it does:** Calls Tavily API for current web evidence. Retries with a broader query if fewer than `_MIN_DISTINCT_DOMAINS` distinct sources are returned.

**Test file:** `tests/test_live_search_tool.py` ✅

```python
from fact_check_agent.src.tools.live_search_tool import _count_distinct_domains, format_search_context

def test_count_distinct_domains():
    results = [
        {"url": "https://bbc.co.uk/news/1"},
        {"url": "https://bbc.co.uk/news/2"},   # duplicate domain
        {"url": "https://reuters.com/article/1"},
    ]
    assert _count_distinct_domains(results) == 2
```

**Toggle to tune:** Change `_MIN_DISTINCT_DOMAINS` in `live_search_tool.py` (default 3) to require more or less source diversity.

---

### 1.3 Cross-Modal Tool (`src/tools/cross_modal_tool.py`)

**What it does:** Makes a single LLM call to check for logical conflicts between the claim text and the image caption. Optionally adds a CLIP similarity score (gated by `ENABLE_CLIP`).

**Test file:** `tests/test_cross_modal_tool.py` ✅

```python
from unittest.mock import patch, MagicMock
from fact_check_agent.src.tools.cross_modal_tool import check_cross_modal

def test_no_image_caption():
    result = check_cross_modal("any claim", None, "fake-key", "gpt-4o")
    assert result == {"flag": False, "explanation": None, "clip_score": None}
```

**Toggle to enable CLIP:** Set `ENABLE_CLIP = True` in `cross_modal_tool.py`. Requires `torch` and `transformers`.

---

### 1.4 Freshness Tool (`src/tools/freshness_tool.py`)

**What it does:** Makes a single LLM call to classify whether a cached verdict needs live re-verification, based on claim text, prior verdict, and days since last verification.

**Test file:** `tests/test_freshness_tool.py` ✅

```python
from unittest.mock import patch
from fact_check_agent.src.tools.freshness_tool import check_freshness
from datetime import datetime, timezone, timedelta

def test_fresh_historical_claim():
    # Historical facts should not trigger revalidation even if old
    with patch("fact_check_agent.src.tools.freshness_tool.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.return_value = ...
        result = check_freshness(
            claim_text="The French Revolution began in 1789.",
            verdict_label="supported",
            verdict_confidence=0.99,
            last_verified_at=datetime.now(timezone.utc) - timedelta(days=365),
            api_key="fake-key",
            model="gpt-4o",
        )
    assert result["revalidate"] is False
```

**Toggle to tune:** Edit `FRESHNESS_CHECK_PROMPT` in `src/prompts.py` — adjust day thresholds per category or add few-shot examples.

---

### 1.5 Verdict Synthesis (LLM prompt in `src/graph/nodes.py`)

**What it does:** Calls gpt-4o with the evidence block and returns a structured verdict JSON. This is a node, not a standalone tool — tested via `test_data_contracts.py`.

**Test file:** `tests/test_data_contracts.py` ✅ (covers prompt rendering, JSON parsing, missing-key fallback, invalid JSON fallback)

```python
from fact_check_agent.src.prompts import VERDICT_SYNTHESIS_PROMPT

def test_verdict_prompt_renders():
    rendered = VERDICT_SYNTHESIS_PROMPT.format(
        claim_text="The moon is made of cheese",
        evidence_block="[LIVE SEARCH] NASA confirms moon is rock.",
        source_credibility_note="Source: nasa.gov",
    )
    assert "moon is made of cheese" in rendered
    assert "evidence_links" in rendered
```

---

## 2. Reflection Agent (`src/agents/reflection_agent.py`)

**What it does:** Maintains per-source, topic-conditioned credibility signals in the `source_credibility` ChromaDB collection. Runs in two directions:

- **READ** (`query_source_credibility`): before verdict synthesis, fetches k nearest (source, topic) observations and returns `{credibility_mean, bias_mean, bias_std, sample_count}` via distance-weighted aggregation.
- **WRITE** (`update_source_credibility`): after verdict synthesis, appends one new observation. Always inserts — never overwrites.

**Credibility signal mapping:**

| Verdict | Signal |
|---|---|
| `supported` (confidence c) | `c / 100` — source made a truthful claim |
| `refuted` (confidence c) | `1 - c / 100` — source made a false claim |
| `misleading` | `0.5` — ambiguous |

**Test file:** `tests/test_reflection_agent.py` ✅

```python
from fact_check_agent.src.agents.reflection_agent import credibility_signal, source_id_from_url

def test_credibility_signal_supported():
    assert credibility_signal("supported", 80) == 0.80

def test_credibility_signal_refuted():
    assert credibility_signal("refuted", 80) == 0.20
```

**Toggle to tune (see also `TODO.md`):**
- Add topic normalisation — strip entities/dates before embedding
- Add recency weighting — `half_life_days` multiplier on distance weights

---

## 3. Routing Logic

### 3.1 Confidence Router (`src/graph/router.py`)

**What it does:** Routes to `freshness_check` if `max_confidence >= 0.80`, else to `live_search`.

**Test file:** `tests/test_router.py` ✅ (covers threshold boundary, None memory, zero confidence)

**Toggle to tune:** Change `CACHE_CONFIDENCE_THRESHOLD` in `router.py`.

| Value | Effect |
|---|---|
| `0.90` | Fewer cache hits, more live search, higher accuracy |
| `0.80` | Default — balanced |
| `0.65` | More cache hits, lower cost, risk of stale verdicts |

### 3.2 Freshness Router (`src/graph/router.py`)

**What it does:** After a cache hit, routes to `return_cached` (fresh) or `live_search` (stale) based on `revalidation_needed` from the freshness tool.

**Test file:** `tests/test_router.py` ❌ `freshness_router` not yet tested

---

## 4. Integration Testing — LangGraph Agent

Test the full agent graph end-to-end with mocked tools and memory — no live API calls required.

**Test file:** `tests/test_graph_integration.py` ✅

Scenarios covered:
- ✅ Live-search path returns a valid `FactCheckOutput`
- ✅ Verdict fields from LLM response land on output correctly
- ✅ Cross-modal flag propagates to output
- ✅ `receive_claim` node resets all mutable state fields
- ❌ Cache path (`route == "cache"`, `return_cached` node exercised)
- ❌ Freshness tool returns `revalidate=True` → graph takes stale path (live search runs)
- ❌ Freshness tool returns `revalidate=False` → graph takes fresh path (live search skipped)
- ❌ `write_memory` called with correct `Verdict` fields
- ❌ `source_credibility` populated in state after `query_memory`
- ❌ Reflection agent `update_source_credibility` called after verdict is written

---

## 5. Benchmark Evaluation

### 5.1 LIAR — Full Test Split

```bash
# Direct eval (no memory seeding)
python -m fact_check_agent.benchmark.run_eval \
    --dataset liar \
    --liar-path /path/to/liar/test.tsv \
    --split test \
    --output results/liar_test.json

# Seed train split first, then eval (exercises cache-hit path)
python -m fact_check_agent.benchmark.run_eval \
    --dataset liar \
    --liar-path /path/to/liar/test.tsv \
    --split test \
    --seed-train /path/to/liar/train.tsv \
    --output results/liar_test_seeded.json
```

Expected baseline: ~35–45% binary accuracy.

### 5.2 FakeNewsNet

```bash
python -m fact_check_agent.benchmark.run_eval \
    --dataset fakenewsnet \
    --fnn-root /path/to/fakenewsnet \
    --source politifact \
    --split test \
    --output results/fnn_politifact.json
```

### 5.3 Reading Results

```python
import json
with open("results/liar_test.json") as f:
    results = json.load(f)

print(f"Macro-F1: {results['macro_f1']:.4f}")

failures = [r for r in results["rows"] if not r["correct"]]
high_conf_wrong = [r for r in failures if r["confidence_score"] > 70]
for r in high_conf_wrong[:5]:
    print(f"  [{r['ground_truth_label']}→{r['predicted_verdict']}] {r['claim_text']}")
```

---

## 6. Prompt Tuning

All prompts are in `src/prompts.py`. Each has a version comment — bump it and re-run benchmarks after any change.

### 6.1 Identify failing classes

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(results["y_true"], results["y_pred"])
print(cm)
```

### 6.2 Tune `VERDICT_SYNTHESIS_PROMPT`

Add few-shot examples after the evidence block:
```python
FEW_SHOT_BLOCK = """
EXAMPLES OF HARD CASES:
- Claim: "GDP grew 3% last quarter" + Evidence: "GDP grew 1.8%" → misleading (exaggerated number)
- Claim: "Senator X voted against bill Y" + Evidence: "Senator X missed the vote" → misleading (framing)
"""
```

The prompt now includes a `source_credibility_note` block populated by the Reflection Agent.
The LLM is instructed to lower its confidence score when source credibility is low and evidence is thin.

### 6.3 Tune `FRESHNESS_CHECK_PROMPT`

Adjust per-category day thresholds or add new categories:
```
- Political claims, election results: re-verify if > 7 days old    ← tune this
- Ongoing events: re-verify if > 3 days old                        ← tune this
- Scientific consensus: re-verify if > 180 days old                ← tune this
```

---

## 7. Enabling SOTA Enhancements

All SOTA features are gated by flags or commented-out code. See also `TODO.md` for planned improvements.

| Enhancement | File | How to enable |
|---|---|---|
| **GraphRAG** | `src/tools/rag_tool.py` | Add `get_entity_claims()` call after `get_entity_context()` in `query_memory` node |
| **Self-RAG** | `src/graph/nodes.py` | Add `IS_RETRIEVAL_NEEDED_PROMPT` call before `live_search` node; filter chunks with `CHUNK_RELEVANCE_PROMPT` |
| **Claim Decomposition** | `src/graph/nodes.py` | Add decomposition node before `live_search`; run synthesis per sub-claim |
| **Multi-Agent Debate** | `src/graph/router.py` | Uncomment the `debate_check` SOTA block (triggers when `35 < confidence < 65`) |
| **CLIP Cross-Modal** | `src/tools/cross_modal_tool.py` | Set `ENABLE_CLIP = True`; requires `torch` and `transformers` |
| **Freshness classifier upgrade** | `src/tools/freshness_tool.py` | Replace single LLM call with a ReAct agent that can check news APIs and entity activity |
| **Topic normalisation** | `src/agents/reflection_agent.py` | Strip entities/dates before embedding claim as topic vector (see `TODO.md`) |
| **Recency weighting** | `src/agents/reflection_agent.py` | Add `half_life_days` multiplier to distance weights using stored `created_at` (see `TODO.md`) |

---

## 8. LangSmith Tracing

With `LANGCHAIN_TRACING_V2=true` in `.env`, every `graph.invoke()` is traced automatically.

**What to inspect:**
- Node execution times — `synthesize_verdict` and `live_search` are typically slowest
- `retrieved_chunks` state — check evidence quality going into the LLM
- `revalidation_needed` state — verify the freshness tool is routing correctly
- `source_credibility` state — verify the reflection agent is populating credibility signals
- Token counts per node — spot over-long prompts

---

## 9. Quick Sanity Check (No API Keys Needed)

```python
# sanity_check.py
from unittest.mock import MagicMock
from fact_check_agent.src.graph.graph import build_graph

memory = MagicMock()
graph  = build_graph(memory)
print("Graph nodes:", list(graph.nodes))
print("Graph compiled successfully.")
```

```bash
PYTHONPATH="" .venv/bin/python sanity_check.py
```
