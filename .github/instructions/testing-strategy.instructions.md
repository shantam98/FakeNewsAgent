---
name: testing-strategy
description: "Test coverage roadmap and testing patterns. Use when: writing new tests, debugging test failures, understanding mock patterns, or filling testing gaps (T3, T4, T5)."
applyTo:
  - "fact_check_agent/tests/**"
  - "fact_check_agent/src/**"
---

# Testing Strategy & Coverage Roadmap

## Current Test Coverage

### ✅ Well-Covered Areas

| Component | Test File | Coverage |
|-----------|-----------|----------|
| Router (debate_check, freshness_router) | test_router.py | ✅ All branches tested |
| Data contracts (field types, formatting) | test_data_contracts.py | ✅ All fields validated |
| RAG tool | test_rag_tool.py | ✅ Search + confidence % |
| Live search tool | test_live_search_tool.py | ✅ Tavily retry + domain counting |
| Freshness tool | test_freshness_tool.py | ✅ Classification logic |
| Cross-modal tool | test_cross_modal_tool.py | ✅ SigLIP/caption conflict |
| Prompts | test_prompts.py | ✅ Template rendering |
| Reranker | test_reranker.py | ✅ RRF merge + cross-encoder |
| Reflection agent | test_reflection_agent.py | ✅ Source credibility queries |

### ⚠️ Testing Gaps (Priority Order)

| ID | Description | File | Impact | Complexity |
|----|-------------|------|--------|-----------|
| **T3** | Cache-hit integration test | test_graph_integration.py | HIGH | Low |
| **T4** | Stale cache path integration test | test_graph_integration.py | HIGH | Low |
| **T5** | Reflection credibility write assertion | test_graph_integration.py | MEDIUM | Medium |

---

## Gap T3: Cache-Hit Integration Test

**What**: Full graph run where memory confidence ≥ 0.80 AND freshness returns fresh → `return_cached` node exercised, live search never called.

**Why**: Validates the happy path where we skip expensive live search due to high-confidence cached verdict.

**Implementation**:

```python
# fact_check_agent/tests/test_graph_integration.py

def test_cache_hit_routes_to_return_cached(make_memory_mock, make_fact_check_input):
    """
    Setup: Memory returns high-confidence (0.95) FRESH verdict.
    Expected: Graph returns cached verdict without calling live_search.
    """
    # 1. Mock memory with high-confidence result
    memory_mock = make_memory_mock(max_confidence=0.95)  # ≥ 0.80 threshold
    
    # Set a recent verified_at (interpreted as "fresh")
    memory_mock.search_similar_claims.return_value = {
        "ids": [["clm_cache001"]],
        "documents": [["Similar claim"]],
        "metadatas": [[{
            "verdict_label": "supported",
            "verdict_confidence": 0.95,
            "verified_at": "2026-04-20T00:00:00Z",  # Recent
        }]],
        "distances": [[0.1]],
    }
    
    # 2. Mock freshness check to return "fresh"
    with patch("fact_check_agent.src.tools.freshness_tool.check_freshness") as mock_freshness:
        mock_freshness.return_value = {
            "verdict_label": "supported",
            "verdict_confidence": 0.95,
            "status": "fresh",
            "revalidation_needed": False,
        }
        
        # 3. Mock live search (should NOT be called)
        with patch("fact_check_agent.src.tools.live_search_tool.search_live") as mock_tavily:
            # 4. Build graph with mocked memory
            from fact_check_agent.src.graph.graph import build_graph
            graph = build_graph(memory_mock)
            
            # 5. Invoke graph
            input_claim = make_fact_check_input(
                claim_text="Test claim",
                image_caption=None,
            )
            result = graph.invoke({
                "input": input_claim,
                "memory_results": memory_mock.search_similar_claims.return_value,
            })
            
            # 6. Assertions
            assert result["output"].verdict == "supported"
            assert result["output"].confidence_score == 95  # From 0.95 * 100
            assert not mock_tavily.called, "Live search should not be called on cache hit"
            assert len(result["context_claims"]) == 0 or all(
                c["from_cache"] for c in result["context_claims"]
            ), "All context should be from cache"
```

**Key Points**:
- Mock `memory.search_similar_claims()` to return high-confidence result
- Set `verified_at` to recent timestamp (interpreted as "fresh")
- Mock `check_freshness()` return to confirm fresh status
- Assert that `live_search_tool.search_live()` is **never called**
- Confidence should be converted correctly: `0.95 → 95`

---

## Gap T4: Stale Cache Path Integration Test

**What**: Cache hit (confidence ≥ 0.80) but freshness returns `revalidate=True` → live search runs, `return_cached` never called.

**Why**: Validates that even high-confidence verdicts are re-checked if old or marked for revalidation.

**Implementation**:

```python
# fact_check_agent/tests/test_graph_integration.py

def test_stale_cache_triggers_live_search(make_memory_mock, make_fact_check_input):
    """
    Setup: Memory returns high-confidence (0.92) but STALE verdict.
    Expected: Graph triggers live_search, ignores cache verdict.
    """
    # 1. Mock memory with high-confidence result
    memory_mock = make_memory_mock(max_confidence=0.92)
    
    # Set old verified_at (will be marked as "stale")
    memory_mock.search_similar_claims.return_value = {
        "ids": [["clm_old001"]],
        "documents": [["Outdated claim"]],
        "metadatas": [[{
            "verdict_label": "supported",
            "verdict_confidence": 0.92,
            "verified_at": "2025-01-01T00:00:00Z",  # Old — over 90 days
        }]],
        "distances": [[0.2]],
    }
    
    # 2. Mock freshness check to return "stale" (revalidate=True)
    with patch("fact_check_agent.src.tools.freshness_tool.check_freshness") as mock_freshness:
        mock_freshness.return_value = {
            "verdict_label": "supported",
            "verdict_confidence": 0.92,
            "status": "stale",
            "revalidation_needed": True,  # ← Key: forces live search
        }
        
        # 3. Mock live search to return fresh evidence
        with patch("fact_check_agent.src.tools.live_search_tool.search_live") as mock_tavily:
            mock_tavily.return_value = [
                {"url": "https://example.com/article1", "content": "New evidence...", "source": "Reuters"},
            ]
            
            # 4. Build graph
            from fact_check_agent.src.graph.graph import build_graph
            graph = build_graph(memory_mock)
            
            # 5. Invoke graph
            input_claim = make_fact_check_input(
                claim_text="Test claim needing revalidation",
                image_caption=None,
            )
            result = graph.invoke({
                "input": input_claim,
                "memory_results": memory_mock.search_similar_claims.return_value,
            })
            
            # 6. Assertions
            assert mock_tavily.called, "Live search MUST be called on stale cache"
            # Verdict may change based on fresh evidence
            assert result["output"] is not None
            assert "https://example.com/article1" in result["output"].evidence_links
```

**Key Points**:
- Mock cache verdict with old `verified_at` (> 90 days)
- Mock `check_freshness()` to return `revalidation_needed=True`
- Mock `search_live()` to return new evidence
- Assert that `search_live()` **is called** (not skipped)
- Verify that fresh evidence is incorporated into verdict

---

## Gap T5: Reflection Agent Credibility Write Assertion

**What**: After a full graph run, assert that `memory.add_source_credibility_point()` was called with correct `source_id` and `point_id`.

**Why**: Validates that source credibility tracking is wired and populated correctly; impacts future verdict weighting.

**Implementation**:

```python
# fact_check_agent/tests/test_graph_integration.py

def test_reflection_credibility_written_to_memory(make_memory_mock, make_fact_check_input):
    """
    Setup: Run full graph end-to-end.
    Expected: Reflection agent writes source credibility update to memory.
    """
    # 1. Create mock memory with tracking
    memory_mock = make_memory_mock(max_confidence=0.0)  # No cache hit → full pipeline
    
    memory_mock.add_source_credibility_point = MagicMock()
    memory_mock.add_verdict = MagicMock()
    memory_mock.get_entity_context.return_value = []
    memory_mock.query_source_credibility.return_value = {}
    
    # 2. Mock LLM to return deterministic verdict
    with patch("fact_check_agent.src.llm_factory.create_client") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm.call.return_value = json.dumps({
            "verdict": "refuted",
            "confidence": 75,
            "reasoning": "Evidence contradicts claim.",
            "bias_score": 0.3,
            "evidence_links": ["https://example.com/evidence"],
        })
        mock_llm_factory.return_value = mock_llm
        
        # 3. Build graph
        from fact_check_agent.src.graph.graph import build_graph
        graph = build_graph(memory_mock)
        
        # 4. Invoke graph with known source
        input_claim = make_fact_check_input(
            claim_text="Test claim",
            source_id="src_reuters",  # Known source
            image_caption=None,
        )
        result = graph.invoke({
            "input": input_claim,
            "memory_results": None,
        })
        
        # 5. Assertions
        assert memory_mock.add_source_credibility_point.called, \
            "Reflection agent must write source credibility"
        
        # Verify the call arguments
        call_args = memory_mock.add_source_credibility_point.call_args
        assert call_args[1]["source_id"] == "src_reuters"
        assert call_args[1]["point_id"].startswith("sc_")  # Format: sc_<verdict_id>
        assert call_args[1]["verdict_label"] == "refuted"
        assert call_args[1]["confidence"] == 0.75  # From 75 / 100
```

**Key Points**:
- Mock both `add_verdict()` and `add_source_credibility_point()` to track calls
- Run full graph end-to-end (no cache hit)
- Assert `add_source_credibility_point()` is called with:
  - `source_id` = input source
  - `point_id` = `f"sc_{verdict_id}"` format
  - `verdict_label` = one of {"supported", "refuted", "misleading"}
  - `confidence` = expressed as 0.0–1.0 (divide by 100 from LLM output)

---

## Common Testing Patterns

### Mock MemoryAgent

```python
# conftest.py
from unittest.mock import MagicMock

@pytest.fixture
def make_memory_mock():
    def _make(max_confidence=0.0):
        """Create a mock MemoryAgent with sensible defaults."""
        memory = MagicMock()
        
        # Default: no results
        memory.search_similar_claims.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        
        # Default: no entity context
        memory.get_entity_context.return_value = []
        
        # Default: no source credibility history
        memory.query_source_credibility.return_value = {}
        
        # Overrideable result
        if max_confidence > 0:
            memory.search_similar_claims.return_value = {
                "ids": [["clm_mock001"]],
                "documents": [["Mock similar claim"]],
                "metadatas": [[{
                    "verdict_label": "supported",
                    "verdict_confidence": max_confidence,
                    "verified_at": "2026-04-20T00:00:00Z",
                }]],
                "distances": [[0.1]],
            }
        
        return memory
    
    return _make
```

### Mock LLM Responses

```python
def make_verdict_response(verdict="supported", confidence=72, **kwargs):
    """Mock OpenAI response for verdict synthesis."""
    response_dict = {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": kwargs.get("reasoning", "Test reasoning."),
        "bias_score": kwargs.get("bias_score", 0.5),
        "evidence_links": kwargs.get("evidence_links", []),
    }
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(response_dict)
    return mock_response
```

### Full Graph Integration Test Template

```python
def test_full_graph_flow(make_memory_mock, make_fact_check_input):
    """Template for end-to-end graph tests."""
    # 1. Setup mocks
    memory_mock = make_memory_mock(max_confidence=0.0)
    
    with patch("fact_check_agent.src.llm_factory.create_client") as mock_llm:
        # 2. Configure LLM behavior
        mock_llm.return_value.call.return_value = json.dumps({...})
        
        # 3. Build graph
        from fact_check_agent.src.graph.graph import build_graph
        graph = build_graph(memory_mock)
        
        # 4. Create input
        claim = make_fact_check_input(claim_text="...", image_caption=None)
        
        # 5. Invoke
        result = graph.invoke({"input": claim})
        
        # 6. Assert
        assert result["output"] is not None
        assert result["output"].verdict in ["supported", "refuted", "misleading"]
```

---

## Test Organization

### Naming Convention

- **Unit tests**: `test_<component>_<behavior>.py`
  - Example: `test_rag_tool.py`, `test_router.py`
- **Integration tests**: `test_graph_integration.py`
- **Contract tests**: `test_data_contracts.py`

### File Structure

```
fact_check_agent/tests/
├── conftest.py                    # Fixtures (make_memory_mock, make_fact_check_input)
├── test_router.py                 # Router edge decisions
├── test_data_contracts.py         # Schema validation
├── test_graph_integration.py      # Full end-to-end flows (includes T3, T4, T5)
├── test_rag_tool.py
├── test_live_search_tool.py
├── test_freshness_tool.py
├── test_cross_modal_tool.py
├── test_prompts.py
├── test_reranker.py
└── test_reflection_agent.py
```

---

## Running Tests

```bash
# All tests
pytest fact_check_agent/tests/ -v

# Single test file
pytest fact_check_agent/tests/test_graph_integration.py -v

# Single test case
pytest fact_check_agent/tests/test_graph_integration.py::test_cache_hit_routes_to_return_cached -v

# With detailed output
pytest fact_check_agent/tests/ -vv

# Skip marked tests
pytest fact_check_agent/tests/ -v -m "not slow"

# Show print statements
pytest fact_check_agent/tests/ -v -s
```

---

## Coverage Goals

| Phase | Target | Status |
|-------|--------|--------|
| Phase 1 (Baseline) | 11 test files, 70% path coverage | ✅ Done |
| Phase 2 (Integration) | Fill T3, T4, T5 gaps | ⚠️ In Progress |
| Phase 3 (SOTA) | Multi-agent debate, GraphRAG paths | 📋 Pending |
| Phase 4 (Edge Cases) | Error handling, DB offline scenarios | 📋 Pending |

---

## Debugging Failed Tests

### 1. Check Logs

```bash
# Logs are written to logs/benchmark_*.log
tail -f logs/benchmark_*.log

# Or view test output
pytest fact_check_agent/tests/test_router.py -vv -s
```

### 2. Isolate the Test

```bash
# Run just the failing test
pytest fact_check_agent/tests/test_graph_integration.py::test_cache_hit_routes_to_return_cached -vv -s

# Add breakpoint in code
import pdb; pdb.set_trace()
```

### 3. Mock Issues

- If a mock isn't working, check the import path
- Verify patch target matches actual import location
- Use `assert mock_obj.called` to verify mock was invoked

### 4. State Issues

- Print state at each node to debug state flow
- Use `pytest -s` to see print output
- Check FactCheckState typing matches returned dict keys

---

## Next Steps

1. **Implement T3** (cache-hit integration test) — validates happy path
2. **Implement T4** (stale cache integration test) — validates revalidation
3. **Implement T5** (reflection credibility write) — validates side effects
4. **Run full suite**: `pytest fact_check_agent/tests/ -v`
5. **Update TODO.md** to mark these tests as done
