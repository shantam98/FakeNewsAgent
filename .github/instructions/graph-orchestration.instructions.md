---
name: graph-orchestration
description: "LangGraph node design and routing for the fact-check agent. Use when: implementing new graph nodes, modifying routing logic, changing state transitions, or debugging the orchestration flow."
applyTo:
  - "fact_check_agent/src/graph/**"
---

# LangGraph Node Implementation Guide

## Node Architecture

All 9 nodes follow this pattern:

```python
def my_node(state: FactCheckState) -> dict[str, Any]:
    """
    Update a specific subset of state fields.
    Return only the fields you modify — LangGraph merges returned dict into state.
    """
    # Read what you need from state
    claim = state["input"]
    memory_results = state["memory_results"]
    
    # Do work
    result = compute_something(claim, memory_results)
    
    # Return ONLY modified fields
    return {"my_field": result}
```

### Key Rules

1. **Pure Functions**: No side effects (except I/O reads). Idempotent.
2. **Minimal Returns**: Return only the fields you modify. LangGraph merges non-conflicting updates.
3. **Type Hints**: Always include `-> dict[str, Any]` return type for clarity.
4. **No Loops**: Nodes are leaf functions. For iterative work (e.g., context_claim_agent), do it inside the node, not via graph edges.

---

## The 9 Nodes (Execution Order)

### 1. receive_claim
**Purpose**: Entry point; initialize state  
**Input**: `input` (FactCheckInput) — set by caller via graph.invoke()  
**Output**: Resets all mutable fields to defaults  
**Example**:
```python
def receive_claim(state: FactCheckState) -> dict[str, Any]:
    return {
        "memory_results": None,
        "entity_context": [],
        "fresh_context": [],
        "stale_context": [],
        "context_claims": [],
        "cross_modal_flag": False,
        "cross_modal_explanation": None,
        "output": None,
    }
```

---

### 2. query_memory
**Purpose**: Semantic search + entity context retrieval  
**Calls**: `memory.search_similar_claims()` + `memory.get_entity_context()`  
**Input**: `input.claim_text`, `input.entities`  
**Output**: `memory_results`, `entity_context`  

**Pattern**:
```python
def query_memory(state: FactCheckState) -> dict[str, Any]:
    memory = get_memory()
    claim_text = state["input"].claim_text
    entities = state["input"].entities
    
    # Also populate source_credibility (read-only from reflection agent)
    # Return BOTH query results AND credibility context
    return {
        "memory_results": memory.search_similar_claims(claim_text),
        "entity_context": memory.get_entity_context(entities),
        "source_credibility": memory.query_source_credibility(entities),  # Read-only
    }
```

---

### 3. freshness_check_all
**Purpose**: Tag memory results as fresh or stale  
**Input**: `memory_results` (list of past verdicts)  
**Output**: `fresh_context`, `stale_context` (split by freshness)  

**Pattern**:
```python
def freshness_check_all(state: FactCheckState) -> dict[str, Any]:
    memory_results = state["memory_results"]
    if not memory_results:
        return {"fresh_context": [], "stale_context": []}
    
    fresh, stale = [], []
    for result in memory_results:
        if is_fresh(result):  # Check age + verdict_label presence
            fresh.append(result)
        else:
            stale.append(result)
    
    return {"fresh_context": fresh, "stale_context": stale}
```

**Freshness Logic**:
- Fresh: verdict_label exists + verified_at recent (< 90 days) + confidence present
- Stale: anything else
- Safe default: stale (triggers live_search as fallback)

---

### 4. context_claim_agent
**Purpose**: Gather evidence to fill gaps  
**Input**: `input`, `fresh_context`, `retrieved_chunks`  
**Output**: `context_claims` (flat list of {type, question, content, source})  

**Pattern** (4-step workflow):
```python
def context_claim_agent_node(state: FactCheckState) -> dict[str, Any]:
    agent = ContextClaimAgent(get_memory())
    
    context_claims = agent.run(
        claim=state["input"],
        fresh_context=state["fresh_context"],
        prefetched_chunks=state.get("retrieved_chunks", []),
    )
    
    return {"context_claims": context_claims}
```

**Internal 4 Steps** (inside ContextClaimAgent.run()):
1. **Q-Gen**: LLM generates 3 factual + 3 counter Q's
2. **Coverage**: LLM checks which Q's are answered by existing context
3. **Search**: For unanswered Q's, call Tavily (or reuse cached chunks)
4. **Summarize**: Each result → {type, question, content, source}

---

### 5. synthesize_verdict
**Purpose**: LLM verdict synthesis with credibility weighting  
**Input**: `input`, `context_claims`, `source_credibility`, `entity_context`  
**Output**: `output` (FactCheckOutput) — fully populated except cross_modal fields  

**Pattern**:
```python
def synthesize_verdict(state: FactCheckState) -> dict[str, Any]:
    verdict_prompt = VERDICT_SYNTHESIS_PROMPT.format(
        claim_text=state["input"].claim_text,
        context_claims=state["context_claims"],
        source_credibility=state.get("source_credibility", {}),
        entity_context=state.get("entity_context", []),
    )
    
    llm = llm_factory.create_client()
    response = llm.call(verdict_prompt)
    
    output = parse_verdict_response(response, state["input"].claim_id)
    return {"output": output}
```

**Output Schema** (FactCheckOutput):
```python
FactCheckOutput(
    verdict_id="vrd_...",
    claim_id=input.claim_id,
    verdict="supported"|"refuted"|"misleading",
    confidence_score=72,  # 0–100 int
    evidence_links=["url1", "url2"],
    reasoning="Chain of thought explanation...",
    bias_score=0.45,
    cross_modal_flag=None,  # Filled by cross_modal_check node
    cross_modal_explanation=None,  # Filled by cross_modal_check node
)
```

---

### 6. debate_check (Router)
**Purpose**: Conditional edge — route low-confidence verdicts to debate or skip  
**Input**: `output.confidence_score`  
**Output**: String "debate" or "skip"  

**Pattern**:
```python
def debate_check(state: FactCheckState) -> str:
    if not settings.use_debate:
        return "skip"
    
    confidence = state["output"].confidence_score
    if confidence < settings.debate_threshold:  # e.g., < 60
        return "debate"
    return "skip"
```

**Wiring** (in graph.py):
```python
graph.add_conditional_edges(
    "synthesize_verdict",
    debate_check,
    {
        "debate": "multi_agent_debate",
        "skip": "cross_modal_check",
    },
)
```

---

### 7. multi_agent_debate (Optional)
**Purpose**: Refine verdict via adversarial debate  
**Enabled**: When `USE_DEBATE=true` AND confidence < threshold  
**Input**: `output` (existing verdict), `context_claims`  
**Output**: `output` (updated with debate_transcript + refined verdict)  

**Pattern**:
```python
def multi_agent_debate(state: FactCheckState) -> dict[str, Any]:
    agent = MultiAgentDebate(get_memory())
    
    updated_output, transcript = agent.run(
        verdict=state["output"],
        context_claims=state["context_claims"],
        claim_text=state["input"].claim_text,
    )
    
    return {
        "output": updated_output,
        "debate_transcript": transcript,
    }
```

**Internal Logic** (3 LLM calls):
1. **Advocate For**: Argue the verdict is correct
2. **Advocate Against**: Argue for opposite conclusion
3. **Arbiter**: Weigh both sides; may flip or refine verdict

---

### 8. cross_modal_check
**Purpose**: Check image-caption consistency with verdict  
**Input**: `input.image_caption`, `output`  
**Output**: `output` (cross_modal_flag + cross_modal_explanation updated)  

**Pattern**:
```python
def cross_modal_check(state: FactCheckState) -> dict[str, Any]:
    output = state["output"]
    image_caption = state["input"].image_caption
    
    if not image_caption:
        output.cross_modal_flag = False
        output.cross_modal_explanation = None
        return {"output": output}
    
    tool = CrossModalTool()
    conflict, explanation = tool.check(
        claim_text=state["input"].claim_text,
        caption=image_caption,
        verdict=output.verdict,
    )
    
    output.cross_modal_flag = conflict
    output.cross_modal_explanation = explanation
    return {"output": output}
```

---

### 9. write_memory
**Purpose**: Persist verdict to Neo4j + ChromaDB; trigger reflection agent  
**Input**: `output`, `input`  
**Output**: None (side effect only)  

**Pattern**:
```python
def write_memory(state: FactCheckState) -> dict[str, Any]:
    memory = get_memory()
    output = state["output"]
    
    # Construct Verdict object (from memory_agent.models.verdict)
    verdict = Verdict(
        verdict_id=output.verdict_id,
        claim_id=output.claim_id,
        verdict_label=output.verdict,
        confidence=output.confidence_score / 100,  # Convert to 0.0–1.0
        reasoning=output.reasoning,
        evidence_summary=f"{output.reasoning}\n\nSources: {', '.join(output.evidence_links)}",
        image_mismatch=output.cross_modal_flag,
        created_at=datetime.now(timezone.utc),
    )
    
    # Write to memory
    memory.add_verdict(verdict)
    
    # Trigger reflection agent (source credibility update)
    source_id = state["input"].source_id or "unknown"
    memory.add_source_credibility_point(
        source_id=source_id,
        point_id=f"sc_{output.verdict_id}",
        verdict_label=output.verdict,
        confidence=output.confidence_score / 100,
    )
    
    return {}  # No state updates
```

**Key Mapping**:
- `confidence`: Divide by 100 (LLM gives 0–100, Verdict expects 0.0–1.0)
- `image_mismatch`: From `cross_modal_flag`
- `evidence_summary`: Combined reasoning + links

---

### 10. emit_output
**Purpose**: Final preparation before return  
**Input**: `output`, `fresh_context`  
**Output**: `output` (with `last_verified_at` if applicable)  

**Pattern**:
```python
def emit_output(state: FactCheckState) -> dict[str, Any]:
    output = state["output"]
    
    # If verdict came from cache (fresh memory), stamp verification time
    if state["fresh_context"] and output.confidence_score >= 80:
        latest_fresh = max(state["fresh_context"], key=lambda x: x.get("verified_at", ""))
        output.last_verified_at = latest_fresh.get("verified_at")
    
    return {"output": output}
```

---

## Routing Decisions

### The Only Router: debate_check()
**Why just one?** The graph is intentionally linear. All evidence is gathered upfront (context_claim_agent), then synthesized once (synthesize_verdict). No backtracking loops or multi-path orchestration.

**Exception handling:**
- Node raises exception → logged + None default returned
- LLM response parse error → fallback to "misleading" verdict
- Missing DB → write_memory fails silently (logged)

All these are safe defaults; graph continues.

---

## Adding a New Node

1. **Understand the input**: What fields do you read? Add those as type hints.
2. **Write the function**: Pure function with minimal side effects.
3. **Return only changed fields**: LangGraph merges returned dict into state.
4. **Add to graph.py**: Call `graph.add_node(name, func)` and wire edges.
5. **Wire edges**: Use `graph.add_edge()` or `add_conditional_edges()`.
6. **Test**: Mock dependencies, call node directly, verify output shape.

**Example: Adding a "fact_checker_filter" node**

```python
# src/graph/nodes.py
def fact_checker_filter(state: FactCheckState) -> dict[str, Any]:
    """Filter context_claims to only high-quality sources."""
    claims = state["context_claims"]
    filtered = [c for c in claims if is_trusted_source(c["source"])]
    return {"context_claims": filtered}

# src/graph/graph.py in build_graph():
graph.add_node("fact_checker_filter", fact_checker_filter)
graph.add_edge("context_claim_agent", "fact_checker_filter")
graph.add_edge("fact_checker_filter", "synthesize_verdict")
```

---

## Debugging a Node

```bash
# Run with detailed logging
python -c "
from fact_check_agent.src.graph.graph import build_graph
from fact_check_agent.src.models.schemas import FactCheckInput

memory = get_memory()
graph = build_graph(memory)

# Set input
state = graph.get_initial_state()
state['input'] = FactCheckInput(...)

# Invoke specific node for testing
from fact_check_agent.src.graph.nodes import query_memory
result = query_memory(state)
print(result)
"
```

Or add breakpoints in test files and run with `-vv` flag.

---

## State Diagram

```
         receive_claim
              | (reset)
              v
         query_memory
              | (search + context)
              v
      freshness_check_all
              | (split fresh/stale)
              v
      context_claim_agent
              | (gather evidence)
              v
      synthesize_verdict
              | (LLM verdict)
              |
         [debate_check router]
         /              \
   debate?          debate=false
      /                    \
     /                      \
 debate_node          cross_modal_check
     |                     |
     +-----+        +------+
           |        |
           v        v
      cross_modal_check
           |
           v
       write_memory
           |
           v
       emit_output
           |
           v
          END
```
