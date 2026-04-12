# Benchmark Datasets — Data Structure & Integration Guide

> **Purpose:** Reference for loading, parsing, and mapping the two public benchmark datasets into the fact-check agent pipeline.
> Both datasets are used to validate the Fact-Check Agent and LangGraph orchestration pipeline against ground truth labels.

---

## Dataset 1 — FakeNewsNet

- **Kaggle:** [mdepak/fakenewsnet](https://www.kaggle.com/datasets/mdepak/fakenewsnet)
- **Source repo:** [KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
- **Paper:** Shu et al., "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media" (2020)
- **Task:** Binary classification — `fake` vs `real`

### 1.1 Sub-datasets

| Sub-dataset | Source | Label basis | Fake articles | Real articles | Total |
|---|---|---|---|---|---|
| **politifact** | PolitiFact fact-checkers | Expert journalist verdicts | ~336 | ~447 | ~783 |
| **gossipcop** | GossipCop entertainment site | Editorial fact-check reviews | ~1,650 | ~16,767 | ~18,417 |
| **Combined** | Both | — | ~2,000 | ~17,200 | ~23,921 |

> GossipCop is heavily imbalanced (~9% fake). Politifact is roughly balanced (~43% fake). Evaluate both separately.

### 1.2 PolitiFact Label → Binary Mapping

PolitiFact uses a 6-point rating scale. FakeNewsNet collapses it to binary:

| PolitiFact fine-grained label | FakeNewsNet binary label |
|---|---|
| `pants-fire` | **fake** |
| `false` | **fake** |
| `barely-true` | **fake** |
| `half-true` | *(excluded or borderline — varies by version)* |
| `mostly-true` | **real** |
| `true` | **real** |

> Note: The exact boundary varies across dataset versions. The Kaggle CSV files encode the final binary label directly in the filename suffix (`_fake.csv` / `_real.csv`) — no further mapping needed at load time.

### 1.3 CSV File Schema (Minimalist Distribution)

Four CSV files are provided:
- `politifact_fake.csv`
- `politifact_real.csv`
- `gossipcop_fake.csv`
- `gossipcop_real.csv`

| Column | Type | Description |
|---|---|---|
| `id` | `str` | Unique article identifier (e.g. `politifact14982`) |
| `url` | `str` | Original source URL of the article |
| `title` | `str` | Headline / title of the article |
| `tweet_ids` | `str` | Tab-separated list of tweet IDs sharing this article |

> **The binary label is not a column** — it is encoded in the filename (`_fake` / `_real`).
> Add a `label` column (`0 = real`, `1 = fake`) when loading.

### 1.4 Full Dataset Folder Structure (Downloaded via Scripts)

```
fakenewsnet_dataset/
├── politifact/
│   ├── fake/
│   │   └── politifact14982/
│   │       ├── news_content.json
│   │       ├── tweets/
│   │       │   └── 1234567890.json       # one file per tweet
│   │       └── retweets/
│   │           └── 1234567890.json
│   └── real/
│       └── politifact15001/
│           ├── news_content.json
│           ├── tweets/
│           └── retweets/
├── gossipcop/
│   ├── fake/  (same structure)
│   └── real/  (same structure)
├── user_profiles/
│   └── <user_id>.json
├── user_timeline_tweets/
│   └── <user_id>.json
├── user_followers/
│   └── <user_id>.json
└── user_following/
    └── <user_id>.json
```

### 1.5 `news_content.json` Schema

This is the primary file for the fact-check pipeline — maps directly onto `memory_agent` models.

```json
{
  "url":          "https://...",
  "title":        "Article headline",
  "text":         "Full article body text",
  "images":       ["https://img1.jpg", "https://img2.jpg"],
  "publish date": "2018-05-14T09:30:00Z"
}
```

| Field | Type | Maps to (`memory_agent`) |
|---|---|---|
| `url` | `str` | `RawArticle.url`, `Article.url` |
| `title` | `str` | `RawArticle.title`, `Article.title` |
| `text` | `str` | `RawArticle.body_text` |
| `images` | `list[str]` | `RawArticle.image_urls` → `ImageCaption.image_url` |
| `publish date` | `str` (ISO 8601) | `RawArticle.published_at` |

### 1.6 Tweet JSON Schema (per-tweet file)

```json
{
  "id_str":         "tweet ID as string",
  "user_id_str":    "user ID as string",
  "created_at":     "Mon May 14 09:30:00 +0000 2018",
  "text":           "tweet text content",
  "favorite_count": 42,
  "retweet_count":  17,
  "in_reply_to_status_id_str": null,
  "geo":            null
}
```

### 1.7 User Profile JSON Schema

```json
{
  "id_str":              "user ID",
  "screen_name":         "handle",
  "created_at":          "account creation date",
  "location":            "...",
  "followers_count":     1234,
  "friends_count":       567,
  "statuses_count":      8910,
  "verified":            false
}
```

### 1.8 Loading FakeNewsNet → `BenchmarkRecord`

See §3.1 for the `BenchmarkRecord` Pydantic model definition.

```python
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

def load_fakenewsnet_article(
    article_dir: Path,
    label: str,    # "fake" | "real"
    source: str,   # "politifact" | "gossipcop"
    split: str,    # "train" | "test" (FakeNewsNet has no official split — caller assigns)
) -> "BenchmarkRecord":
    content   = json.loads((article_dir / "news_content.json").read_text())
    body_text = content.get("text", "")
    title     = content.get("title", "")
    url       = content.get("url", "")
    images    = content.get("images", [])
    pub_raw   = content.get("publish date")

    content_hash = hashlib.sha256(
        f"{url}{title}{body_text[:200]}".encode()
    ).hexdigest()

    return BenchmarkRecord(
        record_id            = f"fnn_{content_hash[:12]}",
        claim_text           = f"{title}. {body_text[:500]}",
        source_url           = url,
        source_domain        = f"{source}.com",
        source_name          = source,
        image_urls           = images,
        image_caption        = None,   # fill via pre-generation step (§4 Step 1)
        article_body         = body_text,
        article_title        = title,
        published_at         = datetime.fromisoformat(pub_raw) if pub_raw else None,
        content_hash         = content_hash,
        ground_truth_label   = label,
        ground_truth_binary  = 1 if label == "fake" else 0,
        dataset              = f"fakenewsnet_{source}",
        split                = split,
    )


def load_fakenewsnet_dataset(
    dataset_root: Path,
    source: str,   # "politifact" | "gossipcop"
    split: str = "test",
) -> list["BenchmarkRecord"]:
    """Walk source/fake and source/real directories, return all BenchmarkRecords."""
    records: list[BenchmarkRecord] = []
    for label in ("fake", "real"):
        label_dir = dataset_root / source / label
        if not label_dir.exists():
            continue
        for article_dir in sorted(label_dir.iterdir()):
            content_file = article_dir / "news_content.json"
            if content_file.exists():
                records.append(
                    load_fakenewsnet_article(article_dir, label, source, split)
                )
    return records
```

### 1.9 Verdict Label Mapping for Evaluation

When the Fact-Check Agent produces a `Verdict`, map it to the binary benchmark label:

| `Verdict.label` | FakeNewsNet binary |
|---|---|
| `"refuted"` | `fake (1)` |
| `"misleading"` | `fake (1)` |
| `"supported"` | `real (0)` |

### 1.10 Evaluation Metrics

- **Primary:** Accuracy, Macro-F1 (required for balanced evaluation on GossipCop)
- **Secondary:** Precision, Recall per class
- **Per-source:** Report PolitiFact and GossipCop results separately
- **Confidence calibration:** Plot `confidence_score` distribution vs. correct/incorrect verdicts

---

## Dataset 2 — LIAR

- **Kaggle:** [doanquanvietnamca/liar-dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset)
- **Paper:** Wang, "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection", ACL 2017 ([arXiv:1705.00648](https://arxiv.org/abs/1705.00648))
- **Task:** 6-way fine-grained truthfulness classification (can also be collapsed to binary or 3-way)

### 2.1 Dataset Statistics

| Split | File | Statements |
|---|---|---|
| Train | `train.tsv` | 10,269 |
| Validation | `valid.tsv` | 1,284 |
| Test | `test.tsv` | 1,267 |
| **Total** | | **12,836** |

> 80 / 10 / 10 split. All statements sourced from PolitiFact.com, collected over ~a decade.

### 2.2 TSV Schema (14 Columns, No Header Row)

| Column Index | Field Name | Type | Description |
|---|---|---|---|
| 0 | `statement_id` | `str` | Unique ID for the statement (e.g. `2635.json`) |
| 1 | `label` | `str` | 6-way truthfulness label (see §2.3) |
| 2 | `statement` | `str` | **The claim text** — primary input to pipeline |
| 3 | `subjects` | `str` | Comma-separated topic(s) (e.g. `"economy,taxes"`) |
| 4 | `speaker` | `str` | Person or entity making the claim |
| 5 | `speaker_job_title` | `str` | Occupation of the speaker at time of statement |
| 6 | `state_info` | `str` | US state or geographic context |
| 7 | `party_affiliation` | `str` | Political party (e.g. `"republican"`, `"democrat"`) |
| 8 | `barely_true_count` | `int` | Speaker's historical count of barely-true statements |
| 9 | `false_count` | `int` | Speaker's historical count of false statements |
| 10 | `half_true_count` | `int` | Speaker's historical count of half-true statements |
| 11 | `mostly_true_count` | `int` | Speaker's historical count of mostly-true statements |
| 12 | `pants_on_fire_count` | `int` | Speaker's historical count of pants-on-fire statements |
| 13 | `context` | `str` | Venue or location of the statement (e.g. `"a television ad"`) |

> **No header row** in the TSV files. Always assign column names manually when loading.

### 2.3 Label Classes and Distribution

| Label | Truthfulness | Approx. Count (train) | Binary mapping | 3-way mapping |
|---|---|---|---|---|
| `pants-fire` | Completely false | ~1,050 | **fake** | **false** |
| `false` | False | ~2,500 | **fake** | **false** |
| `barely-true` | Mostly false | ~2,100 | **fake** | **mixed** |
| `half-true` | Half true | ~2,600 | *(borderline)* | **mixed** |
| `mostly-true` | Mostly true | ~2,500 | **real** | **true** |
| `true` | Completely true | ~2,100 | **real** | **true** |

> For binary evaluation: `{pants-fire, false, barely-true}` → `fake`; `{mostly-true, true}` → `real`; `half-true` either excluded or treated as fake depending on convention.
> The Notion task board highlights **"Half-True" / "Mostly True" as the hardest edge cases** — this is where prompt tuning effort should focus.

### 2.4 Speaker Credit History (Columns 8–12)

These five integer columns encode the **speaker's prior truthfulness track record** on PolitiFact — cumulative counts including the current statement.

They serve as a structured proxy for **source credibility** and map directly onto the HITL credibility graph:

```python
# Derived credibility score from credit history
def speaker_credibility(row) -> float:
    total = row.barely_true_count + row.false_count + row.half_true_count \
          + row.mostly_true_count + row.pants_on_fire_count
    if total == 0:
        return 0.5
    honest = row.mostly_true_count + row.half_true_count * 0.5
    return honest / total  # 0.0–1.0
```

This is directly analogous to `Source.base_credibility` in the memory agent.

### 2.5 Loading LIAR → `BenchmarkRecord`

See §3.1 for the `BenchmarkRecord` Pydantic model definition.

```python
import hashlib
import pandas as pd

LIAR_COLUMNS = [
    "statement_id", "label", "statement", "subjects",
    "speaker", "speaker_job_title", "state_info", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_on_fire_count", "context",
]

def load_liar_dataset(path: str, split: str) -> list["BenchmarkRecord"]:
    """Load a LIAR TSV file and return a list of BenchmarkRecords.

    Args:
        path:  Path to train.tsv / valid.tsv / test.tsv
        split: "train" | "valid" | "test"
    """
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
    records: list[BenchmarkRecord] = []

    for _, row in df.iterrows():
        cred   = speaker_credibility(row)
        stmt_id = str(row.statement_id)
        speaker  = str(row.speaker) if pd.notna(row.speaker) else "unknown"

        records.append(BenchmarkRecord(
            record_id           = f"liar_{stmt_id}",
            claim_text          = str(row.statement),
            source_url          = (
                f"https://www.politifact.com/personalities/"
                f"{speaker.lower().replace(' ', '-')}/"
            ),
            source_domain       = "politifact.com",
            source_name         = speaker,
            image_urls          = [],
            image_caption       = None,   # LIAR has no images — always None
            article_body        = None,   # statement IS the claim; no article body
            article_title       = str(row.statement),
            published_at        = None,   # LIAR has no timestamps
            content_hash        = hashlib.sha256(
                str(row.statement).encode()
            ).hexdigest(),
            ground_truth_label  = str(row.label),
            ground_truth_binary = (
                1 if str(row.label) in ("pants-fire", "false", "barely-true") else 0
            ),
            dataset             = "liar",
            split               = split,
            # LIAR-specific fields
            speaker             = speaker,
            speaker_job_title   = str(row.speaker_job_title) if pd.notna(row.speaker_job_title) else None,
            party_affiliation   = str(row.party_affiliation) if pd.notna(row.party_affiliation) else None,
            subjects            = str(row.subjects) if pd.notna(row.subjects) else None,
            context             = str(row.context) if pd.notna(row.context) else None,
            speaker_credibility = cred,
        ))

    return records
```

### 2.6 Verdict Label Mapping for Evaluation

| `Verdict.label` | LIAR binary | LIAR 6-way (approximate) |
|---|---|---|
| `"refuted"` | `fake (1)` | `false`, `pants-fire` |
| `"misleading"` | `fake (1)` | `barely-true`, `half-true` |
| `"supported"` | `real (0)` | `mostly-true`, `true` |

### 2.7 Evaluation Metrics

- **Primary:** Macro-F1 across 6 classes (the standard benchmark metric from the original paper)
- **Secondary:** Binary accuracy (fake vs. real), per-class precision/recall
- **Hardest classes:** `half-true` and `mostly-true` — focus prompt tuning here (Task 6)
- **Baseline (Wang 2017 paper):** Hybrid CNN+meta-data achieved ~27% accuracy on 6-way; recent LLM-based approaches reach ~40–50%

---

## 3. Shared Integration Patterns

### 3.1 `BenchmarkRecord` — Unified Pydantic Model

`BenchmarkRecord` is the single typed container for a record from either dataset. It holds the raw dataset fields **and** exposes two adapter methods that convert it into the exact types the pipeline consumes. No dicts anywhere.

```python
# benchmark/record.py
import hashlib
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

# ── memory_agent imports (add memory_agent/ to PYTHONPATH) ─────────────────
from src.models.article import Article, Source
from src.models.caption import ImageCaption
from src.models.claim import Claim
from src.models.pipeline import PreprocessingOutput

# ── fact_check_agent imports ────────────────────────────────────────────────
from fact_check_agent.src.models.schemas import EntityRef, FactCheckInput
from fact_check_agent.src.pipeline import claim_to_fact_check_input  # real pipeline bridge


class BenchmarkRecord(BaseModel):
    """Unified record wrapping one item from either FakeNewsNet or LIAR.

    Constructed by load_fakenewsnet_article() or load_liar_dataset().
    Consumed by .to_preprocessing_output() and .to_fact_check_input().
    """

    # ── Core fields (both datasets) ─────────────────────────────────────────
    record_id: str                      # e.g. "liar_2635.json" | "fnn_a1b2c3d4e5f6"
    claim_text: str                     # primary text fed to the fact-check agent
    source_url: str                     # article URL or constructed PolitiFact URL
    source_domain: str                  # e.g. "politifact.com", "gossipcop.com"
    source_name: str                    # human-readable source name or speaker name
    image_urls: list[str] = Field(default_factory=list)
    image_caption: Optional[str] = None  # pre-generated VLM caption; None if no image
    article_body: Optional[str] = None   # full article text (FakeNewsNet); None for LIAR
    article_title: str = ""             # headline; for LIAR this equals claim_text
    published_at: Optional[datetime] = None
    content_hash: str = ""              # SHA-256 of url+title+body[:200]

    # ── Ground truth ─────────────────────────────────────────────────────────
    ground_truth_label: str            # dataset-native: "fake"/"real" or LIAR 6-way
    ground_truth_binary: int           # 0 = real/true, 1 = fake/false

    # ── Provenance ───────────────────────────────────────────────────────────
    dataset: str    # "fakenewsnet_politifact" | "fakenewsnet_gossipcop" | "liar"
    split: str      # "train" | "valid" | "test"

    # ── LIAR-specific (None for FakeNewsNet) ─────────────────────────────────
    speaker: Optional[str] = None
    speaker_job_title: Optional[str] = None
    party_affiliation: Optional[str] = None
    subjects: Optional[str] = None        # comma-separated topics
    context: Optional[str] = None         # venue/location of statement
    speaker_credibility: Optional[float] = None  # derived from credit history cols

    # ────────────────────────────────────────────────────────────────────────
    # Adapter 1: → PreprocessingOutput
    #
    # Use this when you want to seed MemoryAgent with benchmark data before
    # running eval — so the graph and vector DB are populated just like in
    # the real pipeline.
    # ────────────────────────────────────────────────────────────────────────
    def to_preprocessing_output(self) -> PreprocessingOutput:
        """Convert to PreprocessingOutput — the type MemoryAgent.ingest_preprocessed() consumes.

        - For LIAR: statement becomes a single Claim; speaker becomes the Source.
        - For FakeNewsNet: article body becomes body_snippet; title becomes claim_text.
        - Entities list is empty (benchmarking bypasses EntityExtractor).
        - ImageCaption is populated only when image_caption is set.
        """
        now = datetime.now(timezone.utc)

        # ── Source ───────────────────────────────────────────────────────────
        # For LIAR: speaker is the source; credibility derived from credit history.
        # For FakeNewsNet: domain looked up against SOURCE_CATEGORIES priors.
        base_credibility = self.speaker_credibility if self.speaker_credibility is not None \
            else _domain_credibility(self.source_domain)

        source = Source(
            source_id        = f"src_{self.source_domain.replace('.', '_')}",
            name             = self.source_name,
            domain           = self.source_domain,
            category         = _domain_category(self.source_domain),
            base_credibility = base_credibility,
        )

        # ── Article ──────────────────────────────────────────────────────────
        article_id  = f"art_{self.record_id}"
        body_snippet = (
            f"{self.article_title}. {(self.article_body or '')[:500]}"
        ).strip(". ")

        article = Article(
            article_id   = article_id,
            title        = self.article_title,
            url          = self.source_url,
            source_id    = source.source_id,
            published_at = self.published_at or now,
            ingested_at  = now,
            content_hash = self.content_hash,
            body_snippet = body_snippet,
        )

        # ── Claim ─────────────────────────────────────────────────────────────
        # Benchmark records carry one pre-isolated claim — skip ClaimIsolator.
        claim = Claim(
            claim_id     = f"clm_{self.record_id}",
            article_id   = article_id,
            claim_text   = self.claim_text,
            claim_type   = None,          # not inferred during benchmarking
            extracted_at = now,
            status       = "pending",
            entities     = [],            # EntityExtractor bypassed; no MentionSentiment
        )

        # ── ImageCaption ──────────────────────────────────────────────────────
        image_caption = None
        if self.image_caption and self.image_urls:
            image_caption = ImageCaption(
                caption_id  = f"cap_{self.record_id}",
                article_id  = article_id,
                image_url   = self.image_urls[0],
                vlm_caption = self.image_caption,
            )

        return PreprocessingOutput(
            source        = source,
            article       = article,
            claims        = [claim],
            image_caption = image_caption,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Adapter 2: → FactCheckInput
    #
    # Use this for the direct eval path — bypasses MemoryAgent seeding and
    # feeds straight into the LangGraph fact-check pipeline.
    # ────────────────────────────────────────────────────────────────────────
    def to_fact_check_input(self) -> FactCheckInput:
        """Convert to FactCheckInput — the type the LangGraph graph.invoke() consumes.

        Uses claim_to_fact_check_input() from pipeline.py when a full
        PreprocessingOutput is available (i.e. after to_preprocessing_output()
        has been called and ingested). For direct benchmark eval, constructs
        FactCheckInput directly with empty entities — the query_memory node
        will populate entity context at runtime via MemoryAgent.get_entity_context().
        """
        return FactCheckInput(
            claim_id      = f"clm_{self.record_id}",
            claim_text    = self.claim_text,
            entities      = [],           # empty: no EntityExtractor run during benchmarking
            source_url    = self.source_url,
            article_id    = f"art_{self.record_id}",
            image_caption = self.image_caption,   # set by pre-generation step or None
            timestamp     = self.published_at or datetime.now(timezone.utc),
        )


# ── Helpers ─────────────────────────────────────────────────────────────────

_CREDIBILITY_MAP: dict[str, float] = {
    "reuters.com":     0.95,
    "apnews.com":      0.95,
    "bbc.co.uk":       0.90,
    "politifact.com":  0.80,
    "gossipcop.com":   0.65,
}

_CATEGORY_MAP: dict[str, str] = {
    "reuters.com":     "wire_service",
    "apnews.com":      "wire_service",
    "bbc.co.uk":       "news_outlet",
    "politifact.com":  "fact_checker",
    "gossipcop.com":   "news_outlet",
}

def _domain_credibility(domain: str) -> float:
    return _CREDIBILITY_MAP.get(domain, 0.50)

def _domain_category(domain: str) -> str:
    return _CATEGORY_MAP.get(domain, "unknown")
```

### 3.2 Mapping to `memory_agent` Models

| `memory_agent` model field | FakeNewsNet source | LIAR source |
|---|---|---|
| `Source.source_id` | `"src_politifact_com"` / `"src_gossipcop_com"` | `"src_politifact_com"` |
| `Source.name` | `"politifact"` / `"gossipcop"` | `row.speaker` |
| `Source.base_credibility` | `_domain_credibility()` lookup | `speaker_credibility()` derived |
| `Source.category` | `_domain_category()` lookup | `"fact_checker"` |
| `Article.title` | `news_content.title` | `row.statement` |
| `Article.body_snippet` | `title + body[:500]` | `row.statement` |
| `Article.url` | `news_content.url` | Constructed PolitiFact URL |
| `Article.published_at` | `news_content.publish date` | `None` → `datetime.now()` |
| `Claim.claim_text` | `title + body[:500]` (pre-isolated) | `row.statement` |
| `Claim.entities` | `[]` (EntityExtractor bypassed) | `[]` (EntityExtractor bypassed) |
| `ImageCaption.vlm_caption` | Pre-generated offline (§4 Step 1) | `None` (no images) |
| `Verdict.image_mismatch` | Set by cross-modal agent | Always `False` |

### 3.3 LIAR Credit History → HITL Credibility Graph

LIAR's 5 credit history columns feed directly into the HITL Neo4j credibility graph from the Notion task board (`source → topic` edges with credibility scores):

```python
# benchmark/seed_hitl_graph.py
# Seeds the Neo4j graph with speaker-topic credibility edges from LIAR train set.
# Run once before benchmarking to populate the HITL layer.

from src.memory.agent import MemoryAgent
from src.config import settings

def seed_speaker_credibility(records: list[BenchmarkRecord]) -> None:
    memory = MemoryAgent(settings)
    try:
        for record in records:
            if record.dataset != "liar" or not record.speaker_credibility:
                continue
            if not record.subjects:
                continue
            for topic in record.subjects.split(","):
                topic = topic.strip()
                if not topic:
                    continue
                # Reuse GraphStore's merge_source to create speaker→topic edges
                memory._graph.merge_source(
                    source_id        = f"spk_{record.source_name.lower().replace(' ', '_')}",
                    name             = record.source_name,
                    domain           = record.source_domain,
                    category         = f"speaker_{record.party_affiliation or 'unknown'}",
                    base_credibility = record.speaker_credibility,
                )
    finally:
        memory.close()

# Example resulting graph:
# (Source: "Barack Obama", credibility=0.72) -[:PUBLISHES]-> articles on topic "economy"
# (Source: "Donald Trump",  credibility=0.31) -[:PUBLISHES]-> articles on topic "immigration"
```

---

## 5. Dataset Comparison

| Property | FakeNewsNet (PolitiFact) | FakeNewsNet (GossipCop) | LIAR |
|---|---|---|---|
| **Domain** | Political news | Entertainment/gossip | Political statements |
| **Scale** | ~783 articles | ~18,417 articles | 12,836 statements |
| **Label type** | Binary (fake/real) | Binary (fake/real) | 6-way truthfulness |
| **Class balance** | ~43% fake | ~9% fake (imbalanced) | Roughly balanced |
| **Has images** | Yes (`news_content.images`) | Yes | No |
| **Has article body** | Yes (`news_content.text`) | Yes | No (statement only) |
| **Has source metadata** | Domain only | Domain only | Speaker, party, job, state |
| **Has speaker history** | No | No | Yes (5 credit history cols) |
| **Has timestamps** | Yes (publish date) | Yes | No |
| **Has social signal** | Yes (tweet_ids) | Yes | No |
| **Cross-modal test** | Yes (Task 5) | Yes (Task 5) | No (skip Task 5) |
| **Primary metric** | Accuracy, Macro-F1 | Accuracy, Macro-F1 | Macro-F1 (6-way) |
| **Hardest cases** | GossipCop imbalance | GossipCop imbalance | half-true, mostly-true |

---

## 4. Benchmark Evaluation Protocol

### Step 1 — Pre-generate image captions (FakeNewsNet only)

Run the `CaptionGenerator` offline on all `image_urls[0]` before eval. Store results and patch them back into each `BenchmarkRecord.image_caption` before the eval loop. This avoids VLM rate-limiting during the benchmark run.

```python
# benchmark/generate_captions.py
from memory_agent.src.preprocessing.caption_generator import CaptionGenerator
from memory_agent.src.config import settings

gen = CaptionGenerator(api_key=settings.openai_api_key, model=settings.llm_model)

for record in records:
    if record.image_urls and record.image_caption is None:
        record.image_caption = gen.generate_caption(record.image_urls[0])
```

### Step 2 — (Optional) Seed MemoryAgent with benchmark records

Only needed if you want the graph/vector DB populated before eval (e.g., to test the cache-hit path of the router). Uses `to_preprocessing_output()`.

```python
from memory_agent.src.memory.agent import MemoryAgent
from memory_agent.src.config import settings

memory = MemoryAgent(settings)
try:
    for record in train_records:   # seed with train split only
        output = record.to_preprocessing_output()
        memory.ingest_preprocessed(output)   # idempotent — deduplicates by content_hash
finally:
    memory.close()
```

### Step 3 — Run the LangGraph pipeline

Uses `to_fact_check_input()` — fully typed, no dicts.

```python
from fact_check_agent.src.graph.graph import fact_check_graph
from fact_check_agent.src.models.schemas import FactCheckOutput

y_true, y_pred = [], []

for record in test_records:
    fact_check_input = record.to_fact_check_input()         # typed FactCheckInput
    state = fact_check_graph.invoke({"input": fact_check_input})
    output: FactCheckOutput = state["output"]

    y_true.append(record.ground_truth_binary)
    y_pred.append(1 if output.verdict in ("refuted", "misleading") else 0)
```

### Step 4 — Compute metrics

```python
from sklearn.metrics import classification_report, f1_score

print(classification_report(y_true, y_pred, target_names=["real", "fake"]))
print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))

# For LIAR 6-way evaluation — map FactCheckOutput.verdict → LIAR label bucket
VERDICT_TO_LIAR = {
    "supported":  "mostly-true",   # best-effort coarse mapping
    "misleading": "barely-true",
    "refuted":    "false",
}
```

### Step 5 — Targeted prompt tuning (Task 6)

After running evals:
1. Identify per-class error matrix — look for `misleading` mis-predicted as `supported`
2. Sample 20–30 hard cases where `confidence_score` was high but verdict was wrong
3. Inspect `state["retrieved_chunks"]` and `state["debate_transcript"]` from LangSmith traces
4. Add hard cases as few-shot examples in `VERDICT_SYNTHESIS_PROMPT`
5. Re-run evals — structural prompt changes should bump Macro-F1 by 2–5%

---

## 6. Required Dependencies

```
# Data loading
pandas>=2.0
scikit-learn>=1.4     # metrics: f1_score, classification_report

# For FakeNewsNet image caption pre-generation (reuse from memory_agent)
openai>=1.40          # CaptionGenerator uses gpt-4o vision
requests>=2.31
Pillow>=10.0
```

---

## Sources

- [KaiDMML/FakeNewsNet — GitHub](https://github.com/KaiDMML/FakeNewsNet)
- [FakeNewsNet on Kaggle (mdepak)](https://www.kaggle.com/datasets/mdepak/fakenewsnet)
- [LIAR Dataset on Kaggle (doanquanvietnamca)](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset)
- [Wang 2017 — "Liar, Liar Pants on Fire" — ACL Anthology](https://aclanthology.org/P17-2067/)
- [LIAR on HuggingFace (ucsbnlp/liar)](https://huggingface.co/datasets/ucsbnlp/liar)
- [FakeNewsNet Paper — Shu et al. 2020](https://www.cs.emory.edu/~kshu5/files/FakeNewsNet_big_data.pdf)
