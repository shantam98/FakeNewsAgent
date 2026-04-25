---
name: agents-and-tools
description: "Tool and agent implementation patterns. Use when: adding new tools, creating agents for evidence gathering or debate, integrating LLM calls, or extending the fact-check pipeline."
applyTo:
  - "fact_check_agent/src/agents/**"
  - "fact_check_agent/src/tools/**"
---

# Tools & Agents Implementation Guide

## Tools vs Agents

| Aspect | Tool | Agent |
|--------|------|-------|
| **Purpose** | Specific I/O task (search, classify, rerank) | Multi-step workflow with decision logic |
| **Example** | `live_search_tool.py` — call Tavily | `context_claim_agent.py` — Q-gen → search → summarize |
| **Scope** | Single responsibility | Orchestrates multiple tools/LLM calls |
| **LLM Calls** | 0–1 per invocation | Multiple (sequential or conditional) |
| **State** | Stateless | May maintain workflow state |
| **Location** | `src/tools/` | `src/agents/` |

---

## Tools: Implementation Template

### Basic Tool Pattern

```python
# src/tools/my_tool.py

from typing import Any, dict
from fact_check_agent.src.config import settings
from fact_check_agent.src.llm_factory import llm_factory


class MyTool:
    """Single-responsibility tool for a specific fact-check task."""
    
    def __init__(self):
        """Initialize any stateless dependencies."""
        self.llm = llm_factory.create_client()  # Lazy init
    
    def run(self, **kwargs) -> dict[str, Any]:
        """
        Execute the tool. Accept keyword arguments matching the tool's purpose.
        Return a dict with standardized output fields.
        """
        # Validate input
        if not kwargs.get("query"):
            raise ValueError("query is required")
        
        # Execute core logic
        result = self._execute(kwargs["query"])
        
        # Normalize output format
        return self._format_output(result)
    
    def _execute(self, query: str) -> Any:
        """Core implementation. May call external APIs or LLM."""
        # Example: LLM call
        prompt = f"Analyze this query: {query}"
        response = self.llm.call(prompt)
        return response
    
    def _format_output(self, result: Any) -> dict[str, Any]:
        """
        Normalize output to standard format.
        Every tool should return a dict with:
        - status: "success" | "error"
        - data: The result (structure varies by tool)
        - error_message: (optional) If status="error"
        """
        # Parse result, ensure correctness
        try:
            data = self._parse(result)
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "data": None,
            }
        
        return {
            "status": "success",
            "data": data,
        }
    
    def _parse(self, raw: str) -> dict:
        """Parse and validate tool-specific output."""
        import json
        return json.loads(raw)
```

### Real Example: live_search_tool

```python
# src/tools/live_search_tool.py

from typing import Any, Optional
import requests
from fact_check_agent.src.config import settings


class LiveSearchTool:
    """Search the web using Tavily API with domain diversity."""
    
    def __init__(self):
        self.api_key = settings.tavily_api_key
        self.base_url = "https://api.tavily.com/search"
        self._min_distinct_domains = 3
    
    def search_live(self, query: str, max_results: int = 8) -> list[dict]:
        """
        Search the web for evidence on a query.
        
        Args:
            query: Factual question to search
            max_results: Max results to return
        
        Returns:
            List of search results: [{url, content, source}, ...]
        """
        # First round: get initial results
        results = self._call_tavily(query, max_results)
        
        # Check domain diversity
        domains = {self._extract_domain(r["url"]) for r in results}
        
        if len(domains) < self._min_distinct_domains:
            # Retry: if too few domains, refine query
            refined_query = self._diversify_query(query)
            retry_results = self._call_tavily(refined_query, max_results)
            results = self._merge_results(results, retry_results)
        
        return results
    
    def _call_tavily(self, query: str, max_results: int) -> list[dict]:
        """Call Tavily API and extract structured results."""
        response = requests.post(
            self.base_url,
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
            },
        )
        response.raise_for_status()
        
        data = response.json()
        return [
            {
                "url": result["url"],
                "content": result["content"],
                "source": result.get("source", "Unknown"),
            }
            for result in data.get("results", [])
        ]
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        return urlparse(url).netloc
    
    def _diversify_query(self, query: str) -> str:
        """Refine query for broader results."""
        # Remove superlatives, add synonyms, broaden scope
        query = query.replace("most", "").replace("only", "")
        return f"{query} evidence"
    
    def _merge_results(self, r1: list, r2: list) -> list:
        """Merge and deduplicate results."""
        seen_urls = {r["url"] for r in r1}
        merged = r1 + [r for r in r2 if r["url"] not in seen_urls]
        return merged
```

### Tool Error Handling Pattern

```python
class MyTool:
    def run(self, **kwargs) -> dict[str, Any]:
        """Graceful error handling."""
        try:
            # Validate
            if not self._validate_input(kwargs):
                return {
                    "status": "error",
                    "error_message": "Invalid input",
                    "data": None,
                }
            
            # Execute
            result = self._execute(**kwargs)
            
            # Return
            return {
                "status": "success",
                "data": result,
            }
        
        except requests.Timeout as e:
            # Log timeout, return safe default
            logger.error(f"Timeout in {self.__class__.__name__}: {e}")
            return {
                "status": "error",
                "error_message": "Request timeout",
                "data": None,
            }
        
        except Exception as e:
            # Log unexpected errors
            logger.exception(f"Unexpected error in {self.__class__.__name__}")
            return {
                "status": "error",
                "error_message": "Internal error",
                "data": None,
            }
```

---

## Agents: Implementation Template

### Basic Agent Pattern

```python
# src/agents/my_agent.py

from typing import Any
from fact_check_agent.src.memory_client import get_memory
from langgraph.graph import Graph


class MyAgent:
    """Multi-step agent for complex fact-checking workflows."""
    
    def __init__(self, memory=None):
        """
        Initialize agent with memory dependency.
        
        Args:
            memory: MemoryAgent singleton (injected for testing)
        """
        self.memory = memory or get_memory()
    
    def run(self, **kwargs) -> Any:
        """
        Execute the agent workflow.
        
        Returns:
            Result of the multi-step process
        """
        # Step 1: Gather inputs
        claim = kwargs["claim"]
        context = kwargs.get("context", [])
        
        # Step 2: First LLM call or tool call
        step1_result = self.step_1(claim)
        
        # Step 3: Conditional branching
        if self.should_proceed(step1_result):
            step2_result = self.step_2(step1_result, context)
        else:
            step2_result = self.fallback()
        
        # Step 4: Finalize
        final_result = self.finalize(step2_result)
        
        return final_result
    
    def step_1(self, claim: str) -> Any:
        """First step in workflow."""
        pass
    
    def step_2(self, prev_result: Any, context: list) -> Any:
        """Conditional step."""
        pass
    
    def should_proceed(self, result: Any) -> bool:
        """Decide whether to proceed to next step."""
        return result is not None
    
    def fallback(self) -> Any:
        """Fallback if condition not met."""
        return None
    
    def finalize(self, result: Any) -> Any:
        """Final processing."""
        return result
```

### Real Example: context_claim_agent

```python
# src/agents/context_claim_agent.py

from typing import Any
from fact_check_agent.src.llm_factory import llm_factory
from fact_check_agent.src.prompts import (
    GENERATE_QUESTIONS_PROMPT,
    CHECK_QUESTION_COVERAGE_PROMPT,
    SUMMARIZE_SEARCH_RESULT_PROMPT,
)


class ContextClaimAgent:
    """
    4-step evidence gathering agent:
    1. Q-Gen: Generate factual + counter-factual questions
    2. Coverage: Check which questions are answered by existing context
    3. Search: For unanswered Q's, call Tavily
    4. Summarize: Convert each search result into a context claim
    """
    
    def __init__(self, memory=None):
        self.memory = memory or get_memory()
        self.llm = llm_factory.create_client()
    
    def run(
        self,
        claim: str,
        fresh_context: list[dict] = [],
        prefetched_chunks: list[str] = [],
    ) -> list[dict]:
        """
        Run all 4 steps and return flat list of context claims.
        
        Returns:
            [
                {"type": "factual", "question": "Q1", "content": "...", "source": "url"},
                {"type": "counter", "question": "Q2", "content": "...", "source": "url"},
            ]
        """
        # Step 1: Generate questions
        questions = self._generate_questions(claim)
        
        # Step 2: Check coverage
        answered, unanswered = self._check_coverage(
            questions, fresh_context, prefetched_chunks
        )
        
        # Step 3: Search for answers
        search_results = self._search_unanswered(unanswered)
        
        # Step 4: Combine and summarize
        context_claims = self._synthesize_claims(
            answered, unanswered, search_results, fresh_context
        )
        
        return context_claims
    
    def _generate_questions(self, claim: str) -> dict[str, list]:
        """LLM step 1: Generate 3 factual + 3 counter-factual Q's."""
        prompt = GENERATE_QUESTIONS_PROMPT.format(claim_text=claim)
        response = self.llm.call(prompt)
        
        # Parse response as JSON
        import json
        parsed = json.loads(response)
        return {
            "factual": parsed.get("factual_questions", []),
            "counter": parsed.get("counter_questions", []),
        }
    
    def _check_coverage(
        self,
        questions: dict,
        fresh_context: list,
        prefetched_chunks: list,
    ) -> tuple[list, list]:
        """LLM step 2: Which Q's are answered by existing context?"""
        prompt = CHECK_QUESTION_COVERAGE_PROMPT.format(
            questions=questions,
            present_context=[
                c.get("claim_text", c) for c in fresh_context
            ] + prefetched_chunks,
        )
        response = self.llm.call(prompt)
        
        import json
        coverage = json.loads(response)
        return coverage.get("answered", []), coverage.get("unanswered", [])
    
    def _search_unanswered(self, unanswered_q: list) -> list[dict]:
        """Tool step 3: Call Tavily for unanswered questions."""
        from fact_check_agent.src.tools.live_search_tool import LiveSearchTool
        
        tool = LiveSearchTool()
        results = []
        
        for question in unanswered_q:
            # Skip if already in prefetch
            search_results = tool.search_live(question, max_results=5)
            results.extend(search_results)
        
        return results
    
    def _synthesize_claims(
        self,
        answered: list,
        unanswered: list,
        search_results: list,
        fresh_context: list,
    ) -> list[dict]:
        """LLM step 4: Convert to flat context claims."""
        context_claims = []
        
        # From answered questions: extract from fresh_context
        for q in answered:
            matching = [c for c in fresh_context if q in str(c)]
            for m in matching:
                context_claims.append({
                    "type": "factual",
                    "question": q,
                    "content": m.get("claim_text", str(m)),
                    "source": m.get("url", "memory"),
                    "from_cache": True,
                })
        
        # From search results: summarize via LLM
        for result in search_results:
            prompt = SUMMARIZE_SEARCH_RESULT_PROMPT.format(
                question=unanswered[len(context_claims) % len(unanswered)],
                content=result["content"],
                url=result["url"],
            )
            summary = self.llm.call(prompt)
            
            context_claims.append({
                "type": "factual",  # Default; could be refined by LLM
                "question": unanswered[len(context_claims) % len(unanswered)],
                "content": summary,
                "source": result["url"],
                "from_cache": False,
            })
        
        return context_claims
```

---

## Dependency Injection for Testing

### Tool Injection Pattern

```python
# In a graph node, pass tool to agent
def my_node(state: FactCheckState) -> dict:
    # In production, tools create their own clients
    live_search = LiveSearchTool()  # Creates Tavily client
    
    # In tests, inject mock
    agent = MyAgent(memory=mock_memory)  # Testing
    result = agent.run(...)
    
    return {"output": result}

# In tests
def test_my_agent_with_mock_tool(monkeypatch):
    mock_tool = MagicMock()
    monkeypatch.setattr(
        "fact_check_agent.src.agents.my_agent.LiveSearchTool",
        lambda: mock_tool
    )
    
    agent = MyAgent(memory=mock_memory)
    result = agent.run(claim="...", context=[])
    
    assert mock_tool.called
```

---

## Adding a New Tool

1. **Define the contract**: What goes in? What comes out?
   ```python
   # Input: query string
   # Output: {"status": "success|error", "data": ...}
   ```

2. **Create `/src/tools/my_tool.py`**:
   ```python
   class MyTool:
       def __init__(self):
           self.llm = llm_factory.create_client()
       
       def run(self, **kwargs) -> dict:
           # Implementation
           pass
   ```

3. **Write tests** in `/tests/test_my_tool.py`:
   ```python
   def test_my_tool_success():
       tool = MyTool()
       result = tool.run(query="test")
       assert result["status"] == "success"
   ```

4. **Integrate into agent or node**:
   ```python
   # In an agent or node
   tool = MyTool()
   result = tool.run(query=question)
   ```

---

## Adding a New Agent

1. **Identify the workflow**: List the steps in plain English
   - Step 1: Generate questions
   - Step 2: Check coverage
   - Step 3: Search
   - Step 4: Synthesize

2. **Create `/src/agents/my_agent.py`**:
   ```python
   class MyAgent:
       def __init__(self, memory=None):
           self.memory = memory or get_memory()
       
       def run(self, **kwargs):
           result_1 = self.step_1(kwargs)
           result_2 = self.step_2(result_1, kwargs)
           return self.finalize(result_2)
   ```

3. **Add to graph node** or **call from another agent**:
   ```python
   # In graph node:
   def my_graph_node(state: FactCheckState) -> dict:
       agent = MyAgent(memory=get_memory())
       result = agent.run(claim=state["input"].claim_text)
       return {"output": result}
   ```

4. **Write integration test**:
   ```python
   def test_my_agent_full_flow(make_memory_mock):
       memory = make_memory_mock()
       agent = MyAgent(memory=memory)
       result = agent.run(claim="test", context=[])
       assert result is not None
   ```

---

## Configuration & Feature Gating

Tools and agents should respect feature flags:

```python
# In tool/agent
from fact_check_agent.src.config import settings

class MyTool:
    def run(self, **kwargs):
        if settings.use_my_feature:
            return self._run_with_feature(kwargs)
        else:
            return self._run_baseline(kwargs)
```

---

## Common Patterns

### LLM Response Parsing

```python
import json
import re

def parse_json_response(response: str) -> dict:
    """Safely parse JSON from LLM output."""
    # Strip markdown (if present)
    if response.startswith("```"):
        response = re.sub(r"```[a-z]*\n?", "", response).rstrip("`")
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback: return raw string
        return {"error": "Parse failed", "raw": response}
```

### Aggregating Multiple Results

```python
def merge_results(results: list[dict]) -> dict:
    """Merge multiple tool/agent results."""
    merged = {}
    for r in results:
        if r.get("status") == "success":
            merged.update(r.get("data", {}))
    
    return merged
```

### Retry Logic

```python
def retry(func, max_retries=3, backoff=1.0):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff ** attempt)
```

---

## Testing tools & Agents

### Unit Test Template

```python
def test_my_tool_success():
    """Tool succeeds with valid input."""
    tool = MyTool()
    result = tool.run(query="test query")
    
    assert result["status"] == "success"
    assert "data" in result

def test_my_tool_error():
    """Tool handles error gracefully."""
    tool = MyTool()
    result = tool.run(query="")  # Empty query
    
    assert result["status"] == "error"
    assert "error_message" in result
```

### Integration Test Template

```python
def test_my_agent_full_flow(make_memory_mock):
    """Agent completes all steps."""
    memory_mock = make_memory_mock(max_confidence=0.5)
    
    agent = MyAgent(memory=memory_mock)
    result = agent.run(
        claim="Test claim",
        context=[{"claim_text": "Context"}],
    )
    
    assert result is not None
    assert len(result) > 0
```

---

## Common Pitfalls

### ❌ Don't

| Mistake | Fix |
|---------|-----|
| Create new MemoryAgent in tool | Use injected singleton: `get_memory()` |
| Hardcode LLM model names | Use `llm_factory.create_client()` for config agnostic |
| Skip error handling | Always catch + return dict with "error" status |
| Return raw exception | Convert to safe error message string |
| Mix tool logic with graph node | Decompose to separate tool class, call from node |
| Forget to test mocked paths | Mock external deps; test both success & error paths |

### ✅ Do

- Keep tools and agents **single responsibility**
- Make all I/O **injectable** (for testing)
- Return **structured dicts** with status + data
- **Log** errors and unexpected states
- Write **unit tests** for individual steps
- Write **integration tests** for end-to-end flows
