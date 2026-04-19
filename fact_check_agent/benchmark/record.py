"""BenchmarkRecord — unified Pydantic model for FakeNewsNet, LIAR, and Factify2 records.

Provides two typed adapters:
  .to_preprocessing_output() → PreprocessingOutput  (seed MemoryAgent)
  .to_fact_check_input()     → FactCheckInput        (direct eval path)

Loaders:
  load_fakenewsnet_article()  — single article from news_content.json
  load_fakenewsnet_dataset()  — all articles in a source/label directory tree
  load_liar_dataset()         — all rows in a LIAR TSV split file
  load_factify2_dataset()     — tab-separated Factify2 CSV (train/val/test)
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

# ── Bootstrap memory_agent path ───────────────────────────────────────────────
from fact_check_agent.src._bootstrap import *  # noqa: F401,F403

from src.models.article import Article, Source
from src.models.caption import ImageCaption
from src.models.claim import Claim
from src.models.pipeline import PreprocessingOutput

from fact_check_agent.src.models.schemas import EntityRef, FactCheckInput

logger = logging.getLogger(__name__)

# ── Credibility priors (mirrors memory_agent SOURCE_CATEGORIES) ───────────────
_CREDIBILITY_MAP: dict[str, float] = {
    "reuters.com": 0.95, "apnews.com": 0.95,
    "bbc.co.uk": 0.90,   "politifact.com": 0.80,
    "gossipcop.com": 0.65,
}
_CATEGORY_MAP: dict[str, str] = {
    "reuters.com": "wire_service",  "apnews.com": "wire_service",
    "bbc.co.uk": "news_outlet",     "politifact.com": "fact_checker",
    "gossipcop.com": "news_outlet",
}

def _domain_credibility(domain: str) -> float:
    return _CREDIBILITY_MAP.get(domain, 0.50)

def _domain_category(domain: str) -> str:
    return _CATEGORY_MAP.get(domain, "unknown")

def speaker_credibility(row) -> float:
    """Derive credibility score from LIAR credit history columns."""
    total = (
        row.barely_true_count + row.false_count + row.half_true_count
        + row.mostly_true_count + row.pants_on_fire_count
    )
    if total == 0:
        return 0.5
    honest = row.mostly_true_count + row.half_true_count * 0.5
    return float(honest / total)


# ── BenchmarkRecord model ─────────────────────────────────────────────────────

class BenchmarkRecord(BaseModel):
    """Unified container for one record from FakeNewsNet or LIAR."""

    # Core — both datasets
    record_id:           str
    claim_text:          str
    source_url:          str
    source_domain:       str
    source_name:         str
    image_urls:          list[str] = Field(default_factory=list)
    image_caption:       Optional[str] = None
    article_body:        Optional[str] = None
    article_title:       str = ""
    published_at:        Optional[datetime] = None
    content_hash:        str = ""

    # Ground truth
    ground_truth_label:  str
    ground_truth_binary: int   # 0 = real/true, 1 = fake/false
    ground_truth_verdict: Optional[str] = None  # "supported" | "misleading" | "refuted" (Factify2 only)

    # Provenance
    dataset: str   # "fakenewsnet_politifact" | "fakenewsnet_gossipcop" | "liar" | "factify2"
    split:   str   # "train" | "valid" | "test"

    # Pre-fetched evidence (Factify2 Option A — bypasses live search)
    prefetched_document: Optional[str] = None

    # LIAR-specific (None for FakeNewsNet)
    speaker:             Optional[str] = None
    speaker_job_title:   Optional[str] = None
    party_affiliation:   Optional[str] = None
    subjects:            Optional[str] = None
    context:             Optional[str] = None
    speaker_credibility: Optional[float] = None

    # ── Adapter 1: → PreprocessingOutput ─────────────────────────────────────

    def to_preprocessing_output(self) -> PreprocessingOutput:
        """Convert to PreprocessingOutput for MemoryAgent.ingest_preprocessed().

        Use to seed the memory store with benchmark records before eval.
        """
        now = datetime.now(timezone.utc)

        base_cred = (
            self.speaker_credibility
            if self.speaker_credibility is not None
            else _domain_credibility(self.source_domain)
        )

        source = Source(
            source_id        = f"src_{self.source_domain.replace('.', '_')}",
            name             = self.source_name,
            domain           = self.source_domain,
            category         = _domain_category(self.source_domain),
            base_credibility = base_cred,
        )

        article_id   = f"art_{self.record_id}"
        body_snippet = f"{self.article_title}. {(self.article_body or '')[:500]}".strip(". ")

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

        claim = Claim(
            claim_id     = f"clm_{self.record_id}",
            article_id   = article_id,
            claim_text   = self.claim_text,
            claim_type   = None,
            extracted_at = now,
            status       = "pending",
            entities     = [],
        )

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

    # ── Adapter 2: → FactCheckInput ───────────────────────────────────────────

    def to_fact_check_input(self) -> FactCheckInput:
        """Convert to FactCheckInput for direct graph.invoke() in eval loop.

        Entities are empty — the query_memory node will populate entity context
        at runtime via MemoryAgent.get_entity_context() if records were seeded.
        """
        prefetched: list[str] = []
        if self.prefetched_document:
            prefetched = [f"[REFERENCE DOCUMENT]\n{self.prefetched_document[:3000]}"]

        return FactCheckInput(
            claim_id          = f"clm_{self.record_id}",
            claim_text        = self.claim_text,
            entities          = [],
            source_url        = self.source_url,
            article_id        = f"art_{self.record_id}",
            image_caption     = self.image_caption,
            timestamp         = self.published_at or datetime.now(timezone.utc),
            prefetched_chunks = prefetched,
        )


# ── FakeNewsNet loaders ───────────────────────────────────────────────────────

def load_fakenewsnet_article(
    article_dir: Path,
    label: str,   # "fake" | "real"
    source: str,  # "politifact" | "gossipcop"
    split: str,
) -> Optional[BenchmarkRecord]:
    content_file = article_dir / "news_content.json"
    if not content_file.exists():
        return None
    try:
        content   = json.loads(content_file.read_text())
    except Exception as e:
        logger.warning("Failed to read %s: %s", content_file, e)
        return None

    body_text = content.get("text", "")
    title     = content.get("title", "")
    url       = content.get("url", "")
    images    = content.get("images", [])
    pub_raw   = content.get("publish date")

    content_hash = hashlib.sha256(
        f"{url}{title}{body_text[:200]}".encode()
    ).hexdigest()

    published_at = None
    if pub_raw:
        try:
            published_at = datetime.fromisoformat(pub_raw)
        except ValueError:
            pass

    return BenchmarkRecord(
        record_id           = f"fnn_{content_hash[:12]}",
        claim_text          = f"{title}. {body_text[:500]}".strip(". "),
        source_url          = url,
        source_domain       = f"{source}.com",
        source_name         = source,
        image_urls          = images,
        image_caption       = None,
        article_body        = body_text,
        article_title       = title,
        published_at        = published_at,
        content_hash        = content_hash,
        ground_truth_label  = label,
        ground_truth_binary = 1 if label == "fake" else 0,
        dataset             = f"fakenewsnet_{source}",
        split               = split,
    )


def load_fakenewsnet_dataset(
    dataset_root: Path,
    source: str,
    split: str = "test",
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for label in ("fake", "real"):
        label_dir = dataset_root / source / label
        if not label_dir.exists():
            continue
        for article_dir in sorted(label_dir.iterdir()):
            record = load_fakenewsnet_article(article_dir, label, source, split)
            if record:
                records.append(record)
    logger.info("Loaded %d FakeNewsNet records (%s)", len(records), source)
    return records


# ── LIAR loaders ─────────────────────────────────────────────────────────────

LIAR_COLUMNS = [
    "statement_id", "label", "statement", "subjects",
    "speaker", "speaker_job_title", "state_info", "party_affiliation",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_on_fire_count", "context",
]


def load_liar_dataset(path: str, split: str) -> list[BenchmarkRecord]:
    """Load a LIAR TSV split file → list[BenchmarkRecord]."""
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
    records: list[BenchmarkRecord] = []

    for _, row in df.iterrows():
        cred    = speaker_credibility(row)
        stmt_id = str(row.statement_id)
        speaker = str(row.speaker) if pd.notna(row.speaker) else "unknown"

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
            image_caption       = None,
            article_body        = None,
            article_title       = str(row.statement),
            published_at        = None,
            content_hash        = hashlib.sha256(str(row.statement).encode()).hexdigest(),
            ground_truth_label  = str(row.label),
            ground_truth_binary = (
                1 if str(row.label) in ("pants-fire", "false", "barely-true") else 0
            ),
            dataset             = "liar",
            split               = split,
            speaker             = speaker,
            speaker_job_title   = str(row.speaker_job_title) if pd.notna(row.speaker_job_title) else None,
            party_affiliation   = str(row.party_affiliation) if pd.notna(row.party_affiliation) else None,
            subjects            = str(row.subjects) if pd.notna(row.subjects) else None,
            context             = str(row.context) if pd.notna(row.context) else None,
            speaker_credibility = cred,
        ))

    logger.info("Loaded %d LIAR records from %s (%s split)", len(records), path, split)
    return records


# ── Factify2 loaders ──────────────────────────────────────────────────────────

_FACTIFY2_VERDICT_MAP: dict[str, str] = {
    "Support_Multimodal":      "supported",
    "Support_Text":            "supported",
    "Insufficient_Multimodal": "misleading",
    "Insufficient_Text":       "misleading",
    "Refute":                  "refuted",
}


def load_factify2_dataset(path: str, split: str) -> list[BenchmarkRecord]:
    """Load a Factify2 tab-separated CSV → list[BenchmarkRecord].

    Ground-truth is the 5-way Category column, mapped to 3-way pipeline verdicts.
    When a reference document is present it is stored in prefetched_document so
    to_fact_check_input() can inject it directly, bypassing the live search node
    (Option A eval — $0 Tavily cost).
    """
    df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
    records: list[BenchmarkRecord] = []

    for _, row in df.iterrows():
        category = str(row.get("Category", "")).strip()
        if not category or category not in _FACTIFY2_VERDICT_MAP:
            continue

        verdict   = _FACTIFY2_VERDICT_MAP[category]
        claim     = str(row.get("claim", "")).strip()
        document  = row.get("document")
        doc_text  = str(document).strip() if pd.notna(document) and str(document).strip() else None

        claim_img = row.get("claim_image")
        img_url   = str(claim_img).strip() if pd.notna(claim_img) and str(claim_img).strip() else ""

        content_hash = hashlib.sha256(claim.encode()).hexdigest()
        record_id    = f"factify2_{split}_{content_hash[:12]}"

        records.append(BenchmarkRecord(
            record_id              = record_id,
            claim_text             = claim,
            source_url             = img_url or "https://factify2.dataset/unknown",
            source_domain          = "factify2.dataset",
            source_name            = "Factify2",
            image_urls             = [img_url] if img_url else [],
            image_caption          = None,
            article_body           = doc_text,
            article_title          = claim[:120],
            published_at           = None,
            content_hash           = content_hash,
            ground_truth_label     = category,
            ground_truth_binary    = 0 if verdict == "supported" else 1,
            ground_truth_verdict   = verdict,
            dataset                = "factify2",
            split                  = split,
            prefetched_document    = doc_text,
        ))

    logger.info("Loaded %d Factify2 records from %s (%s split)", len(records), path, split)
    return records
