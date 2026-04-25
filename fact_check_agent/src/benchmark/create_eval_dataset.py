"""Create stratified 6k evaluation dataset from Factify2.

6 buckets (1k each, except modal_refuted which is capped at available rows):
  text_supported    — Category=Support_Text                             (no image needed)
  text_misleading   — Category=Insufficient_Text                        (no image needed)
  text_refuted      — Category=Refute, claim_image NOT locally cached   (no image needed)
  modal_supported   — Category=Support_Multimodal, image locally cached
  modal_misleading  — Category=Insufficient_Multimodal, image locally cached
  modal_refuted     — Category=Refute, image locally cached

Steps:
  1. Load train+val, assign rows to buckets
  2. Extract multi-label topics per claim via nemotron-3-nano:4b (cached)
  3. Normalize vocabulary — LLM merges synonyms into canonical tags
  4. Weighted stratified sample (inverse topic frequency weighting)
  5. Save eval_6k.csv + topic_vocabulary.json

Usage:
    python -m fact_check_agent.src.benchmark.create_eval_dataset [OPTIONS]

    --workers N     Parallel Ollama threads (default: 8)
    --pool N        Max rows per bucket to extract topics for (default: 3000)
    --target N      Target rows per bucket (default: 1000)
    --seed N        Random seed (default: 42)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATASET_ROOT = (
    Path(__file__).resolve().parents[3] / "datasets" / "Factify2" / "Factify 2"
)
MAPPING_PATH  = DATASET_ROOT.parent / "url_to_local.json"
OUT_DIR       = DATASET_ROOT.parents[1] / "eval_dataset"
CACHE_PATH    = OUT_DIR / "topic_cache.json"
VOCAB_PATH    = OUT_DIR / "topic_vocabulary.json"
OUT_CSV       = OUT_DIR / "eval_6k.csv"

SPLIT_PATHS = {
    "train": DATASET_ROOT / "factify2_train" / "factify2" / "train.csv",
    "val":   DATASET_ROOT / "factify2_train" / "factify2" / "val.csv",
}

BUCKET_DEFS = {
    "text_supported":   {"category": "Support_Text",            "needs_image": False},
    "text_misleading":  {"category": "Insufficient_Text",       "needs_image": False},
    "text_refuted":     {"category": "Refute",                  "needs_image": False, "image_present": False},
    "modal_supported":  {"category": "Support_Multimodal",      "needs_image": True},
    "modal_misleading": {"category": "Insufficient_Multimodal", "needs_image": True},
    "modal_refuted":    {"category": "Refute",                  "needs_image": True,  "image_present": True},
}

BUCKET_LABEL = {
    "text_supported":   "supported",
    "text_misleading":  "misleading",
    "text_refuted":     "refuted",
    "modal_supported":  "supported",
    "modal_misleading": "misleading",
    "modal_refuted":    "refuted",
}

TOPIC_PROMPT = """\
You are a topic tagger. Given a list of factual claims, extract 2-5 topic tags for each.
Tags should be lowercase, short (1-3 words), and general enough to apply to many claims.
Good examples: "us politics", "climate change", "covid-19", "immigration", "gun control",
"economy", "healthcare", "sports", "technology", "military", "crime", "election fraud",
"foreign policy", "education", "racial justice", "religion", "media", "science".

Return ONLY a JSON array of arrays (one inner array per claim, in the same order). No explanation.

Claims:
{claims}"""

NORMALIZE_PROMPT = """\
Below is a list of topic tags extracted from news fact-check claims.
Merge any synonymous or near-duplicate tags into a single canonical form.
Rules:
- Prefer shorter, more general forms ("us politics" over "american political affairs")
- Merge plural/singular ("elections" → "elections")
- Merge obvious variants ("covid" / "covid-19" / "coronavirus" → "covid-19")
- Keep distinct topics separate
- Target 40-80 canonical topics total

Return a JSON object mapping EVERY original tag to its canonical form.
Tags that are already canonical should map to themselves.

Tags:
{tags}"""


BATCH_SIZE = 10  # claims per Ollama call


def _ollama_client() -> OpenAI:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return OpenAI(base_url=base_url, api_key="ollama")


def _claim_key(claim: str) -> str:
    return hashlib.md5(claim.encode()).hexdigest()


def _extract_topics_batch(claims: list[str], model: str, client: OpenAI) -> list[list[str]]:
    """Extract topics for a batch of claims in one Ollama call."""
    numbered = "\n".join(f"{i+1}. {c.strip()}" for i, c in enumerate(claims))
    prompt = TOPIC_PROMPT.format(claims=numbered)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=BATCH_SIZE * 60,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)
        if isinstance(result, list) and len(result) == len(claims):
            return [
                [str(t).lower().strip() for t in row if t]
                if isinstance(row, list) else []
                for row in result
            ]
    except Exception as e:
        logger.debug("Batch topic extraction failed: %s", e)
    return [[] for _ in claims]


def extract_topics(claims: list[str], model: str, workers: int) -> dict[str, list[str]]:
    """Extract topics for all claims in batches. Returns {claim_key: [topics]}. Uses cache."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cache: dict[str, list[str]] = {}
    if CACHE_PATH.exists():
        with CACHE_PATH.open() as f:
            cache = json.load(f)

    pending = [c for c in claims if _claim_key(c) not in cache]
    batches = [pending[i: i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    print(f"\nTopic extraction: {len(claims)} total, {len(cache)} cached, "
          f"{len(pending)} pending → {len(batches)} batches of {BATCH_SIZE}")

    if not batches:
        return cache

    client = _ollama_client()
    new_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_extract_topics_batch, batch, model, client): batch
            for batch in batches
        }
        with tqdm(total=len(pending), desc="Topics", unit="claim") as bar:
            for future in as_completed(futures):
                batch = futures[future]
                results = future.result()
                for claim, topics in zip(batch, results):
                    cache[_claim_key(claim)] = topics
                new_count += len(batch)
                bar.update(len(batch))
                if new_count % 500 == 0:
                    with CACHE_PATH.open("w") as f:
                        json.dump(cache, f)

    with CACHE_PATH.open("w") as f:
        json.dump(cache, f)
    print(f"  Saved {new_count} new topic extractions to cache")
    return cache


def normalize_vocabulary(raw_tags: set[str], model: str) -> dict[str, str]:
    """Ask LLM to merge synonymous topic tags. Returns {raw: canonical}."""
    print(f"\nNormalizing vocabulary ({len(raw_tags)} unique raw tags)...")
    tags_str = json.dumps(sorted(raw_tags), indent=2)
    client = _ollama_client()

    # Split into chunks if too many tags
    tag_list = sorted(raw_tags)
    CHUNK = 150
    mapping: dict[str, str] = {}

    for i in range(0, len(tag_list), CHUNK):
        chunk = tag_list[i: i + CHUNK]
        prompt = NORMALIZE_PROMPT.format(tags=json.dumps(chunk, indent=2))
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            chunk_map = json.loads(raw)
            mapping.update(chunk_map)
        except Exception as e:
            logger.warning("Normalization chunk failed: %s — using identity mapping", e)
            mapping.update({t: t for t in chunk})

    # Identity fallback for any missing tags
    for t in raw_tags:
        if t not in mapping:
            mapping[t] = t

    print(f"  {len(raw_tags)} raw tags → {len(set(mapping.values()))} canonical topics")
    return mapping


def load_buckets(mapping: dict) -> dict[str, pd.DataFrame]:
    """Load train+val, assign image availability, split into 6 buckets."""
    frames = []
    for split, path in SPLIT_PATHS.items():
        if not path.exists():
            logger.warning("Split path not found: %s", path)
            continue
        df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
        df = df.dropna(subset=["claim", "document"])
        df = df[df["claim"].str.strip() != ""]
        df["split"] = split
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["image_local"] = df["claim_image"].apply(
        lambda u: str(u).strip() in mapping if pd.notna(u) else False
    )

    buckets: dict[str, pd.DataFrame] = {}
    for name, defn in BUCKET_DEFS.items():
        mask = df["Category"] == defn["category"]
        if "image_present" in defn:
            mask &= df["image_local"] == defn["image_present"]
        buckets[name] = df[mask].copy().reset_index(drop=True)
        print(f"  {name:25s}: {len(buckets[name])} rows")

    return buckets


def stratified_sample(
    df: pd.DataFrame,
    topic_lists: list[list[str]],
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Weighted sampling by inverse topic frequency — rare topics get priority."""
    from collections import Counter

    freq = Counter(t for topics in topic_lists for t in topics)

    weights = np.array([
        sum(1.0 / freq[t] for t in topics) if topics else 1.0
        for topics in topic_lists
    ], dtype=float)

    # Cap at available rows
    n = min(n, len(df))
    weights /= weights.sum()

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(df), size=n, replace=False, p=weights)
    return df.iloc[sorted(chosen)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified 6k eval dataset from Factify2.")
    parser.add_argument("--model",   default="nemotron-3-nano:4b",
                        help="Ollama model for topic extraction (default: nemotron-3-nano:4b)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for topic extraction (default: 8)")
    parser.add_argument("--pool",    type=int, default=3000,
                        help="Max rows per bucket to extract topics for (default: 3000)")
    parser.add_argument("--target",  type=int, default=1000,
                        help="Target rows per bucket in final dataset (default: 1000)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    # 1. Load URL mapping and assign buckets
    url_map: dict = {}
    if MAPPING_PATH.exists():
        with MAPPING_PATH.open() as f:
            url_map = json.load(f)
    print(f"Loaded {len(url_map)} local image mappings")

    print("\nLoading buckets...")
    buckets = load_buckets(url_map)

    # 2. Build candidate pool per bucket (cap at --pool rows)
    print(f"\nBuilding candidate pools (max {args.pool} per bucket)...")
    pools: dict[str, pd.DataFrame] = {}
    rng = random.Random(args.seed)
    for name, df in buckets.items():
        if len(df) > args.pool:
            idx = rng.sample(range(len(df)), args.pool)
            pools[name] = df.iloc[sorted(idx)].reset_index(drop=True)
        else:
            pools[name] = df
        print(f"  {name:25s}: {len(pools[name])} in pool")

    # 3. Extract topics for all pool rows
    all_claims = list({
        row["claim"]
        for df in pools.values()
        for _, row in df.iterrows()
    })
    topic_cache = extract_topics(all_claims, args.model, args.workers)

    # 4. Attach raw topics to each pool row
    for name, df in pools.items():
        df["raw_topics"] = df["claim"].apply(
            lambda c: topic_cache.get(_claim_key(c), [])
        )

    # 5. Normalize vocabulary
    all_raw = {t for df in pools.values() for topics in df["raw_topics"] for t in topics}
    vocab_map = normalize_vocabulary(all_raw, args.model)

    # Apply canonical mapping
    def canonicalize(raw_topics: list[str]) -> list[str]:
        seen = set()
        result = []
        for t in raw_topics:
            canon = vocab_map.get(t, t)
            if canon not in seen:
                seen.add(canon)
                result.append(canon)
        return result

    for name, df in pools.items():
        df["topics"] = df["raw_topics"].apply(canonicalize)

    # 6. Stratified sample per bucket
    print(f"\nSampling {args.target} rows per bucket...")
    sampled_parts: list[pd.DataFrame] = []
    for name, df in pools.items():
        topic_lists = df["topics"].tolist()
        sample = stratified_sample(df, topic_lists, args.target, args.seed)
        sample["bucket"] = name
        sample["verdict_label"] = BUCKET_LABEL[name]
        sampled_parts.append(sample)
        print(f"  {name:25s}: {len(sample)} sampled")

    # 7. Save outputs
    result = pd.concat(sampled_parts, ignore_index=True)

    # Replace image URLs with local paths where available
    result["claim_image"] = result["claim_image"].apply(
        lambda u: url_map.get(str(u).strip(), str(u).strip()) if pd.notna(u) else u
    )

    # Compute topic vocabulary with frequencies
    from collections import Counter
    topic_freq = Counter(
        t for topics in result["topics"] for t in topics
    )
    vocab_out = [{"topic": t, "count": c} for t, c in topic_freq.most_common()]

    result.to_csv(OUT_CSV, sep="\t", index=False)
    with VOCAB_PATH.open("w") as f:
        json.dump(vocab_out, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Eval dataset saved to : {OUT_CSV}")
    print(f"Topic vocabulary saved: {VOCAB_PATH}")
    print(f"Total rows            : {len(result)}")
    print(f"Unique canonical topics: {len(topic_freq)}")
    print(f"\nTopic distribution (top 20):")
    for item in vocab_out[:20]:
        print(f"  {item['topic']:30s}: {item['count']}")


if __name__ == "__main__":
    main()
