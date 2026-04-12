# Memory Agent Codebase — Summary for Fact-Check Agent Integration

> **Purpose:** This document summarises the existing `memory_agent/` codebase so the Fact-Check Agent team can consume it correctly without modifying it.

---

## 1. High-Level Architecture

The `memory_agent/` package runs a three-stage pipeline:

```
ScraperAgent → PreprocessingAgent → MemoryAgent
```

- **ScraperAgent** — fetches raw articles from multiple sources (Tavily Search, RSS feeds, Telegram).
- **PreprocessingAgent** — transforms raw articles into structured claims, entities, and image captions using LLM calls.
- **MemoryAgent** — the single public facade; writes structured data into two databases and exposes query methods to all other agents.

The entry point is `src/pipeline.py:run_pipeline()`, which wires all three stages together.

---

## 2. Storage Backends

| Store | Technology | Purpose |
|---|---|---|
| **Vector DB** | ChromaDB (Cloud or local) | Semantic similarity search over claims, articles, verdicts, image captions |
| **Knowledge Graph** | Neo4j Aura | Entity relationships, claim history, credibility tracking, predictions |

The Fact-Check Agent **never talks to VectorStore or GraphStore directly** — it only calls `MemoryAgent` methods.

---

## 3. Data Models (Pydantic)

### `RawArticle` (`src/scraper/fetchers/base.py`)
Intermediate format between Scraper and Preprocessing. Not relevant to Fact-Check Agent.

```python
@dataclass
class RawArticle:
    url: str
    title: str
    body_text: str
    image_urls: list[str]
    source_name: str
    source_domain: str
    published_at: Optional[datetime]
    content_hash: str          # SHA-256, set after hashing
```

### `Source` (`src/models/article.py`)
```python
class Source(BaseModel):
    source_id: str
    name: str
    domain: str
    category: str              # "wire_service", "social_media", "news_outlet", "unknown"
    base_credibility: float    # 0.0–1.0
```

### `Article` (`src/models/article.py`)
```python
class Article(BaseModel):
    article_id: str            # prefixed ID e.g. "art_..."
    title: str
    url: str
    source_id: str
    published_at: datetime
    ingested_at: datetime
    content_hash: str          # SHA-256
    body_snippet: str          # title + first 500 chars of body
```

### `MentionSentiment` (`src/models/claim.py`)
Edge data on the Claim→Entity relationship.
```python
class MentionSentiment(BaseModel):
    entity_id: str
    name: str
    entity_type: str           # "person", "organization", "country", "location", "event", "product"
    sentiment: str             # "positive", "negative", "neutral"
```

### `Claim` (`src/models/claim.py`)
```python
class Claim(BaseModel):
    claim_id: str              # prefixed ID e.g. "clm_..."
    article_id: str
    claim_text: str
    claim_type: Optional[str]  # "statistical", "attribution", "causal", "predictive"
    extracted_at: datetime
    status: str                # "pending" | "verified" | "expired"
    entities: list[MentionSentiment]
```

### `ImageCaption` (`src/models/caption.py`)
```python
class ImageCaption(BaseModel):
    caption_id: str            # prefixed ID e.g. "cap_..."
    article_id: str
    image_url: str
    vlm_caption: str           # Objective physical description from VLM — no subjective language
```

### `Verdict` (`src/models/verdict.py`)
**Written by the Fact-Check Agent.** This is the output model the Fact-Check Agent must produce.
```python
class Verdict(BaseModel):
    verdict_id: str
    claim_id: str
    label: str                 # "supported" | "refuted" | "misleading"
    confidence: float          # 0.0–1.0 (map to 0–100 in API responses)
    evidence_summary: str
    bias_score: float          # 0.0–1.0
    image_mismatch: bool       # True if cross-modal inconsistency detected
    verified_at: datetime
```

### `PreprocessingOutput` (`src/models/pipeline.py`)
The contract object `MemoryAgent.ingest_preprocessed()` consumes. The Fact-Check Agent receives the `claim` and `image_caption` fields from this.
```python
class PreprocessingOutput(BaseModel):
    source: Source
    article: Article
    claims: list[Claim]
    image_caption: Optional[ImageCaption]
```

### `CredibilitySnapshot` and `Prediction` (`src/models/credibility.py`)
Used by Entity Tracker Agent and Prediction Agent — not directly needed by Fact-Check Agent.

---

## 4. MemoryAgent Public API (`src/memory/agent.py`)

The Fact-Check Agent imports and calls only these methods:

### Write Methods

| Method | Signature | Description |
|---|---|---|
| `add_verdict` | `(verdict: Verdict) -> None` | Writes a fact-check verdict to both ChromaDB and Neo4j. Call after every fact-check. |

### Read / Query Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `search_similar_claims` | `(text: str, top_k: int = 5)` | `dict` (ChromaDB result) | Semantic search for similar past claims by claim text |
| `get_claims_by_ids` | `(ids: list[str])` | `dict` | Fetch claims by exact IDs |
| `get_caption_by_article` | `(article_id: str)` | `dict` | Get VLM image caption for an article |
| `get_verdict_by_claim` | `(claim_id: str)` | `dict` | Check if a claim already has a verdict (cache check) |
| `get_entity_context` | `(claim_id: str)` | `list[dict]` | Get all entities in a claim with their current credibility scores — key for GraphRAG |
| `get_entity_claims` | `(entity_id: str, since: Optional[datetime])` | `list[dict]` | Historical claims and verdicts for an entity — key for GraphRAG |
| `get_source_credibility` | `(article_id: str)` | `Optional[float]` | Source credibility score (0–1) |
| `get_trending_entities` | `(since: datetime, limit: int)` | `list[dict]` | Entities with high recent mention counts |

### ChromaDB Query Result Shape
`search_similar_claims` and similar methods return standard ChromaDB dict format:
```python
{
    "ids": [["clm_abc", "clm_xyz"]],
    "documents": [["claim text 1", "claim text 2"]],
    "distances": [[0.12, 0.34]],
    "metadatas": [[{"article_id": ..., "source_id": ..., "status": ...}, ...]]
}
```

---

## 5. Graph Schema (Neo4j)

**Node types:**
- `Source` → `Article` via `PUBLISHES`
- `Article` → `Claim` via `CONTAINS`
- `Article` → `ImageCaption` via `HAS_IMAGE`
- `Claim` → `Entity` via `MENTIONS {sentiment}`
- `Claim` → `Verdict` via `VERIFIED_AS`
- `Entity` → `CredibilitySnapshot` via `TRACKED_OVER_TIME`
- `Entity` → `Prediction` via `SUBJECT_OF`

**GraphRAG traversal pattern** (1–2 hop from claim):
```
(Claim)-[:MENTIONS]->(Entity)<-[:MENTIONS]-(OtherClaim)-[:VERIFIED_AS]->(Verdict)
```

---

## 6. Configuration (`src/config.py`)

All settings loaded from `.env` via `pydantic-settings`:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM and embeddings (gpt-4o, text-embedding-3-small) |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Neo4j Aura |
| `CHROMA_API_KEY` / `CHROMA_TENANT` / `CHROMA_DATABASE` | ChromaDB Cloud |
| `TAVILY_API_KEY` | Tavily Search API (used by both ScraperAgent and Fact-Check Agent live search) |
| `TELEGRAM_SCRAPER_API_URL` / `TELEGRAM_SCRAPER_API_KEY` | Optional Telegram fetcher |
| `EMBEDDING_MODEL` | Default: `text-embedding-3-small` |
| `LLM_MODEL` | Default: `gpt-4o` |

---

## 7. Preprocessing LLM Prompts (Read-Only Reference)

Located at `src/preprocessing/prompts.py`. The Fact-Check Agent should **reference** these for consistency but **not import** from this module.

- **`CLAIM_ISOLATION_PROMPT`** — extracts falsifiable claims from article text; classifies type as `statistical | attribution | causal | predictive`
- **`ENTITY_EXTRACTION_BATCH_PROMPT`** — refines NER candidates into canonical entities with type and sentiment per-claim
- **`CAPTION_PROMPT`** — instructs VLM to produce objective, physical-only image descriptions (no subjective language, no context inference)

---

## 8. Key Conventions

- All IDs are prefixed: `art_`, `clm_`, `cap_`, `src_`, `vrd_` (use `src/id_utils.py:make_id()`)
- Claim `status` lifecycle: `"pending"` → `"verified"` (set by `create_verdict`) or `"expired"`
- `image_mismatch` on `Verdict` maps directly to `cross_modal_flag` in the API output contract
- `confidence` on `Verdict` is stored as `float 0.0–1.0`; the API output contract uses `0–100` int — convert at the boundary
- Source credibility priors (hardcoded in `PreprocessingAgent`): BBC=0.90, Reuters=0.95, AP=0.95, Telegram=0.30, unknown=0.50
- Image captions are **always objective and physical** — the prompt explicitly forbids subjective language
