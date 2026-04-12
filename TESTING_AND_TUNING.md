# Testing & Tuning Guide — Fact-Check Agent

> How to test each component in isolation, run end-to-end evaluation, and tune the pipeline to improve Macro-F1.

---

## 0. Setup

```bash
cd fact_check_agent
pip install -r requirements.txt
cp .env.example .env          # fill in API keys
export PYTHONPATH=$(pwd)/..   # makes both fact_check_agent and memory_agent importable
```

---

## 1. Unit Testing Individual Components

Each agent module is a pure Python function — testable without a live LLM or database.

### 1.1 RAG Agent

**What to test:** Correct parsing of ChromaDB result dicts, graceful handling of empty results, context block formatting.

```python
# tests/test_rag_agent.py
from fact_check_agent.src.agents.rag_agent import format_rag_context

def test_format_rag_context_with_verdict():
    claims = [{
        "claim_id": "clm_1", "claim_text": "Vaccines cause autism",
        "verdict_label": "refuted", "verdict_confidence": 0.95, "distance": 0.12,
    }]
    result = format_rag_context(claims)
    assert "refuted" in result
    assert "95%" in result

def test_format_rag_context_empty():
    result = format_rag_context([])
    assert "No similar claims" in result

# Mock MemoryAgent for retrieve_similar_claims
class MockMemory:
    def search_similar_claims(self, text, top_k):
        return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}
    def get_verdict_by_claim(self, claim_id):
        return {"metadatas": []}

def test_retrieve_empty_memory():
    from fact_check_agent.src.agents.rag_agent import retrieve_similar_claims
    results = retrieve_similar_claims("any claim", MockMemory())
    assert results == []
```

**Toggle to test:** Change `top_k` in `retrieve_similar_claims()` to see how many results are returned.

---

### 1.2 Live Search Agent

**What to test:** Domain deduplication logic, fallback retry query, formatting of Tavily results.

```python
# tests/test_live_search_agent.py
from fact_check_agent.src.agents.live_search_agent import (
    _count_distinct_domains, format_search_context
)

def test_count_distinct_domains():
    results = [
        {"url": "https://bbc.co.uk/news/1"},
        {"url": "https://bbc.co.uk/news/2"},   # duplicate domain
        {"url": "https://reuters.com/article/1"},
    ]
    assert _count_distinct_domains(results) == 2

def test_format_search_context_empty():
    context, links = format_search_context([])
    assert "No results" in context
    assert links == []

def test_format_search_context():
    results = [{"url": "https://example.com", "title": "Test", "content": "Some content"}]
    context, links = format_search_context(results)
    assert "example.com" in context
    assert links == ["https://example.com"]
```

**Toggle to tune:** Change `_MIN_DISTINCT_DOMAINS` in `live_search_agent.py` (default 3) to require more source diversity.

---

### 1.3 Cross-Modal Agent

**What to test:** Returns `flag=False` when `image_caption=None`, handles LLM JSON parse errors gracefully.

```python
# tests/test_cross_modal_agent.py
from unittest.mock import patch, MagicMock
from fact_check_agent.src.agents.cross_modal_agent import check_cross_modal

def test_no_image_caption():
    result = check_cross_modal("any claim", None, "fake-key", "gpt-4o")
    assert result == {"flag": False, "explanation": None, "clip_score": None}

def test_llm_detects_conflict():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"conflict": true, "explanation": "Image shows peace rally, claim says riot."}'
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        result = check_cross_modal("Violent riot erupted", "People holding peace signs", "k", "gpt-4o")
    assert result["flag"] is True
    assert "peace" in result["explanation"]
```

**Toggle to enable CLIP:** Set `ENABLE_CLIP = True` in `cross_modal_agent.py`. Requires `torch` and `transformers`.

---

### 1.4 Router

**What to test:** Confidence threshold boundary, handles missing/None memory_results.

```python
# tests/test_router.py
from fact_check_agent.src.graph.router import router, CACHE_CONFIDENCE_THRESHOLD
from fact_check_agent.src.models.schemas import MemoryQueryResponse, SimilarClaim

def make_state(max_confidence):
    return {
        "memory_results": MemoryQueryResponse(results=[], max_confidence=max_confidence),
    }

def test_router_cache_hit():
    assert router(make_state(0.85)) == "cache"

def test_router_live_search():
    assert router(make_state(0.75)) == "live_search"

def test_router_exact_threshold():
    # At exactly 0.80 → cache
    assert router(make_state(CACHE_CONFIDENCE_THRESHOLD)) == "cache"

def test_router_no_memory():
    assert router({"memory_results": None}) == "live_search"
```

**Toggle to tune:** Change `CACHE_CONFIDENCE_THRESHOLD` in `router.py`. Lower = more cache hits, less live search cost.

---

### 1.5 Verdict Synthesis (LLM prompt)

**What to test:** Prompt renders correctly, JSON output parses into `FactCheckOutput`.

```python
# tests/test_prompts.py
from fact_check_agent.src.prompts import VERDICT_SYNTHESIS_PROMPT

def test_verdict_prompt_renders():
    rendered = VERDICT_SYNTHESIS_PROMPT.format(
        claim_text="The moon is made of cheese",
        evidence_block="[LIVE SEARCH] NASA confirms moon is rock.",
        source_credibility_note="Source: nasa.gov (credibility: 0.98)",
    )
    assert "moon is made of cheese" in rendered
    assert "evidence_links" in rendered   # JSON key present in template
```

**LLM output validation** — after calling the real API:
```python
import json

def validate_verdict_json(raw: str):
    result = json.loads(raw)
    assert result["verdict"] in ("supported", "refuted", "misleading")
    assert 0 <= result["confidence_score"] <= 100
    assert 0.0 <= result["bias_score"] <= 1.0
    assert isinstance(result["reasoning"], str)
    assert isinstance(result["evidence_links"], list)
```

---

## 2. Integration Testing — LangGraph Graph

Test the full graph end-to-end with a mock MemoryAgent, no live API calls required.

```python
# tests/test_graph_integration.py
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fact_check_agent.src.graph.graph import build_graph
from fact_check_agent.src.models.schemas import (
    FactCheckInput, MemoryQueryResponse, SimilarClaim
)


def make_mock_memory(max_confidence=0.0):
    """Mock MemoryAgent that returns controllable responses."""
    memory = MagicMock()
    memory.search_similar_claims.return_value = {
        "ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]
    }
    memory.get_entity_context.return_value = []
    memory.get_source_credibility.return_value = 0.7
    memory.get_caption_by_article.return_value = {"documents": []}
    memory.get_verdict_by_claim.return_value = {"metadatas": []}
    memory.add_verdict.return_value = None
    return memory


def make_input(claim_text="Test claim", image_caption=None):
    return FactCheckInput(
        claim_id="clm_test_001",
        claim_text=claim_text,
        entities=[],
        source_url="https://example.com",
        article_id="art_test_001",
        image_caption=image_caption,
        timestamp=datetime.now(timezone.utc),
    )


def test_graph_live_search_path():
    """Full graph run on the live-search path with mocked LLM + memory."""
    mock_memory = make_mock_memory(max_confidence=0.0)   # forces live_search

    mock_llm_verdict = '{"verdict": "refuted", "confidence_score": 78, "bias_score": 0.3, "reasoning": "Test reasoning.", "evidence_links": ["https://fact.com"]}'
    mock_llm_cross   = '{"conflict": false, "explanation": null}'

    with patch("openai.OpenAI") as mock_openai, \
         patch("fact_check_agent.src.agents.live_search_agent.TavilyClient") as mock_tavily:

        mock_tavily.return_value.search.return_value = {"results": [
            {"url": "https://reuters.com/1", "title": "Test", "content": "Some evidence"},
            {"url": "https://bbc.co.uk/1",  "title": "Test", "content": "More evidence"},
            {"url": "https://apnews.com/1", "title": "Test", "content": "Extra evidence"},
        ]}
        mock_openai.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=mock_llm_verdict))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=mock_llm_cross))]),
        ]

        graph = build_graph(mock_memory)
        state = graph.invoke({"input": make_input()})

    output = state["output"]
    assert output is not None
    assert output.verdict == "refuted"
    assert output.confidence_score == 78
    assert output.cross_modal_flag is False
    mock_memory.add_verdict.assert_called_once()


def test_graph_cache_path():
    """Graph should return_cached when memory confidence > 0.80."""
    mock_memory = make_mock_memory()
    # Simulate a high-confidence memory hit
    mock_memory.search_similar_claims.return_value = {
        "ids":       [["clm_existing"]],
        "documents": [["Prior claim about same topic"]],
        "distances": [[0.05]],
        "metadatas": [[{"article_id": "art_1", "source_id": "src_1", "status": "verified"}]],
    }
    mock_memory.get_verdict_by_claim.return_value = {
        "metadatas": [{"label": "supported", "confidence": 0.92}]
    }

    mock_llm_verdict = '{"verdict": "supported", "confidence_score": 92, "bias_score": 0.1, "reasoning": "Cached.", "evidence_links": []}'
    mock_llm_cross   = '{"conflict": false, "explanation": null}'

    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=mock_llm_verdict))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=mock_llm_cross))]),
        ]
        graph = build_graph(mock_memory)
        state = graph.invoke({"input": make_input()})

    assert state["route"] == "cache"
    assert state["output"].verdict == "supported"


def test_graph_cross_modal_flagged():
    """Cross-modal flag should propagate to FactCheckOutput."""
    mock_memory = make_mock_memory()
    mock_llm_verdict = '{"verdict": "misleading", "confidence_score": 60, "bias_score": 0.6, "reasoning": "Out of context.", "evidence_links": []}'
    mock_llm_cross   = '{"conflict": true, "explanation": "Image shows a hospital, claim mentions a protest."}'

    with patch("openai.OpenAI") as mock_openai, \
         patch("fact_check_agent.src.agents.live_search_agent.TavilyClient") as mock_tavily:

        mock_tavily.return_value.search.return_value = {"results": [
            {"url": f"https://source{i}.com", "title": "T", "content": "C"} for i in range(3)
        ]}
        mock_openai.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=mock_llm_verdict))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=mock_llm_cross))]),
        ]

        graph = build_graph(mock_memory)
        state = graph.invoke({"input": make_input(image_caption="People in hospital beds")})

    assert state["output"].cross_modal_flag is True
    assert "hospital" in state["output"].cross_modal_explanation
```

---

## 3. Benchmark Evaluation

### 3.1 LIAR — Full Test Split

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

Expected baseline: ~35–45% binary accuracy. SOTA LLM approaches reach ~50%.

### 3.2 FakeNewsNet

```bash
# Generate captions first (one-time, cached to pickle)
python -m fact_check_agent.benchmark.generate_captions \
    --dataset-root /path/to/fakenewsnet \
    --source politifact \
    --cache politifact_captions.pkl

# Run eval
python -m fact_check_agent.benchmark.run_eval \
    --dataset fakenewsnet \
    --fnn-root /path/to/fakenewsnet \
    --source politifact \
    --split test \
    --output results/fnn_politifact.json
```

### 3.3 Reading Results

```python
import json
from sklearn.metrics import confusion_matrix

with open("results/liar_test.json") as f:
    results = json.load(f)

print(f"Macro-F1: {results['macro_f1']:.4f}")

# Identify hardest failures
failures = [r for r in results["rows"] if not r["correct"]]
high_conf_wrong = [r for r in failures if r["confidence_score"] > 70]
print(f"\nHigh-confidence wrong predictions: {len(high_conf_wrong)}")
for r in high_conf_wrong[:5]:
    print(f"  [{r['ground_truth_label']}→{r['predicted_verdict']}] {r['claim_text']}")
```

---

## 4. Prompt Tuning

All prompts are in [src/prompts.py](fact_check_agent/src/prompts.py). Tuning workflow:

### 4.1 Identify failing classes

```python
from sklearn.metrics import confusion_matrix
import numpy as np

cm = confusion_matrix(results["y_true"], results["y_pred"])
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
```

Common failure: `misleading` predicted as `supported` (model is too generous).

### 4.2 Add few-shot examples to `VERDICT_SYNTHESIS_PROMPT`

```python
# Add to the prompt template after the evidence block:
FEW_SHOT_BLOCK = """
EXAMPLES OF HARD CASES:
- Claim: "GDP grew 3% last quarter" + Evidence: "GDP grew 1.8%" → misleading (exaggerated number)
- Claim: "Senator X voted against bill Y" + Evidence: "Senator X missed the vote" → misleading (framing)
"""
```

Append `FEW_SHOT_BLOCK` to `VERDICT_SYNTHESIS_PROMPT` and re-run eval. Track Macro-F1 delta.

### 4.3 Adjust confidence threshold for cache hits

| `CACHE_CONFIDENCE_THRESHOLD` | Effect |
|---|---|
| `0.90` | Fewer cache hits, more live search, higher accuracy, more cost |
| `0.80` | Default — balanced |
| `0.65` | More cache hits, lower cost, may reuse stale verdicts |

Change in [src/graph/router.py](fact_check_agent/src/graph/router.py).

### 4.4 Adjust minimum source diversity for live search

Change `_MIN_DISTINCT_DOMAINS` in [src/agents/live_search_agent.py](fact_check_agent/src/agents/live_search_agent.py).

| Value | Effect |
|---|---|
| `1` | Accept any result — lower latency, less diverse evidence |
| `3` | Default — 3 distinct domains required |
| `5` | High diversity — more retries, higher cost |

---

## 5. Enabling SOTA Enhancements

All SOTA features are gated by flags or commented-out code. None affect the baseline.

| Enhancement | File | How to enable |
|---|---|---|
| **GraphRAG** | `src/agents/rag_agent.py` | Add `get_entity_claims()` call after `get_entity_context()` in `query_memory` node |
| **Self-RAG** | `src/graph/nodes.py` | Add `IS_RETRIEVAL_NEEDED_PROMPT` call before `live_search` node; filter chunks with `CHUNK_RELEVANCE_PROMPT` before `synthesize_verdict` |
| **Claim Decomposition** | `src/graph/nodes.py` | Add decomposition node before `live_search`; run synthesis per sub-claim |
| **Multi-Agent Debate** | `src/graph/router.py` | Uncomment the `debate_check` SOTA block (triggers when `35 < confidence < 65`) |
| **CLIP Cross-Modal** | `src/agents/cross_modal_agent.py` | Set `ENABLE_CLIP = True`; requires `torch` and `transformers` |

Enable one at a time and re-run benchmarks after each — measure the Macro-F1 delta before combining.

---

## 6. LangSmith Tracing

With `LANGCHAIN_TRACING_V2=true` in `.env`, every `graph.invoke()` is traced automatically.

**What to inspect in LangSmith:**
- Node execution times — identify slow nodes (usually `synthesize_verdict` and `live_search`)
- `retrieved_chunks` state value — check evidence quality going into the LLM
- `debate_transcript` — review debate quality when SOTA debate is enabled
- Token counts per node — spot over-long prompts

**Filtering useful runs:**
```python
# LangSmith Python SDK — find high-confidence wrong predictions
from langsmith import Client
client = Client()

runs = client.list_runs(project_name="fakenews-factcheck")
for run in runs:
    output = run.outputs.get("output", {})
    if output.get("confidence_score", 0) > 70:
        print(run.id, output.get("verdict"), output.get("claim_id"))
```

---

## 7. Quick Sanity Check (No API Keys Needed)

Verify the graph compiles and all imports resolve before setting up full credentials:

```python
# sanity_check.py
from unittest.mock import MagicMock
from fact_check_agent.src.graph.graph import build_graph

memory = MagicMock()
graph  = build_graph(memory)
print("Graph nodes:", list(graph.nodes))
print("Graph compiled successfully.")
```

Run with:
```bash
cd fakenews
PYTHONPATH=. python sanity_check.py
```
