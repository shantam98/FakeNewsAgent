"""Topic-stratified 3k dataset from eval_6k.csv.

Uses gpt-oss:20b-cloud to classify each claim into one of 6 broad topics,
then samples 500 per topic bucket → 3k output.

Usage:
    python -m fact_check_agent.src.benchmark.stratify_topics [OPTIONS]

    --model       Ollama model (default: gpt-oss:20b-cloud)
    --workers N   Parallel threads (default: 6)
    --target N    Rows per topic bucket (default: 500)
    --seed N      Random seed (default: 42)
    --output      Output CSV path (default: datasets/eval_dataset/eval_3k.csv)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).resolve().parents[3]
EVAL_DIR   = REPO_ROOT / "datasets" / "eval_dataset"
INPUT_CSV  = EVAL_DIR / "eval_6k.csv"
CACHE_PATH = EVAL_DIR / "topic_class_cache.json"

TOPICS = [
    "politics",
    "health",
    "crime",
    "military",
    "society",
    "other",
]

TOPIC_PROMPT = """\
Classify the following news claim into EXACTLY ONE of these topic categories:
  politics   — elections, politicians, government, legislation, political parties
  health     — COVID-19, vaccines, medicine, disease, healthcare, public health
  crime      — crimes, murder, arrests, court cases, terrorism, law enforcement
  military   — wars, armed conflicts, defense, weapons, military operations
  society    — protests, religion, environment, natural disasters, culture, education
  other      — sports, technology, economy, entertainment, celebrity, science, viral content

Claim: {claim}

Return ONLY one word — the category name. No explanation, no punctuation."""

BATCH_PROMPT = """\
Classify each of the following numbered news claims into EXACTLY ONE of these topic categories:
  politics   — elections, politicians, government, legislation, political parties
  health     — COVID-19, vaccines, medicine, disease, healthcare, public health
  crime      — crimes, murder, arrests, court cases, terrorism, law enforcement
  military   — wars, armed conflicts, defense, weapons, military operations
  society    — protests, religion, environment, natural disasters, culture, education
  other      — sports, technology, economy, entertainment, celebrity, science, viral content

Return a JSON array with one category string per claim, in the same order.
Valid values: "politics", "health", "crime", "military", "society", "other"

Claims:
{claims}"""

BATCH_SIZE = 15


def _client(base_url: str = "http://localhost:11434/v1") -> OpenAI:
    return OpenAI(base_url=base_url, api_key="ollama")


def _key(claim: str) -> str:
    return hashlib.md5(claim.encode()).hexdigest()


def _classify_batch(claims: list[str], model: str, client: OpenAI) -> list[str]:
    numbered = "\n".join(f"{i+1}. {c.strip()[:300]}" for i, c in enumerate(claims))
    prompt = BATCH_PROMPT.format(claims=numbered)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # No max_tokens — reasoning models need to finish thinking before producing content
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)
        if isinstance(result, list) and len(result) == len(claims):
            cleaned = []
            for v in result:
                v = str(v).lower().strip().strip('"').strip("'")
                cleaned.append(v if v in TOPICS else "other")
            return cleaned
    except Exception:
        pass
    # Fallback: classify individually
    out = []
    for claim in claims:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": TOPIC_PROMPT.format(claim=claim[:300])}],
                temperature=0,
            )
            v = resp.choices[0].message.content.strip().lower().strip('"').strip("'")
            out.append(v if v in TOPICS else "other")
        except Exception:
            out.append("other")
    return out


def classify_all(claims: list[str], model: str, workers: int) -> dict[str, str]:
    """Classify all claims; returns {claim_key: topic}. Uses cache."""
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        with CACHE_PATH.open() as f:
            cache = json.load(f)

    pending = [c for c in claims if _key(c) not in cache]
    batches = [pending[i: i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    print(f"\nTopic classification: {len(claims)} total, {len(cache)} cached, "
          f"{len(pending)} pending → {len(batches)} batches")

    if not batches:
        return cache

    client = _client()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_classify_batch, b, model, client): b for b in batches}
        with tqdm(total=len(pending), desc="Classifying", unit="claim") as bar:
            for future in as_completed(futures):
                batch = futures[future]
                results = future.result()
                for claim, topic in zip(batch, results):
                    cache[_key(claim)] = topic
                done += len(batch)
                bar.update(len(batch))
                if done % 300 == 0:
                    with CACHE_PATH.open("w") as f:
                        json.dump(cache, f)

    with CACHE_PATH.open("w") as f:
        json.dump(cache, f)
    print(f"  Classification complete — cache saved")
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="gpt-oss:20b-cloud")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--target",  type=int, default=500)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  default=str(EVAL_DIR / "eval_3k.csv"))
    args = parser.parse_args()

    df = pd.read_csv(INPUT_CSV, sep="\t")
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    claims = df["claim"].tolist()
    cache = classify_all(claims, args.model, args.workers)

    df["topic"] = df["claim"].apply(lambda c: cache.get(_key(c), "other"))

    print("\nTopic distribution (full 6k):")
    dist = df["topic"].value_counts()
    for topic, count in dist.items():
        status = "OK" if count >= args.target else f"WARN — only {count} rows"
        print(f"  {topic:15s}: {count:5d}  {status}")

    # Sample target per topic, balancing verdict_label within each bucket
    parts = []
    for topic in TOPICS:
        bucket = df[df["topic"] == topic]
        n = min(args.target, len(bucket))
        if "verdict_label" in bucket.columns:
            # Try to balance verdict types
            sampled = (
                bucket.groupby("verdict_label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), n // 3 + 1), random_state=args.seed))
            )
            if len(sampled) > n:
                sampled = sampled.sample(n, random_state=args.seed)
            elif len(sampled) < n:
                remaining = bucket.drop(sampled.index)
                extra = remaining.sample(min(n - len(sampled), len(remaining)),
                                         random_state=args.seed)
                sampled = pd.concat([sampled, extra])
        else:
            sampled = bucket.sample(n, random_state=args.seed)
        parts.append(sampled)
        print(f"  {topic:15s}: sampled {len(sampled)}")

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle

    out_path = Path(args.output)
    out.to_csv(out_path, sep="\t", index=False)

    print(f"\n{'='*55}")
    print(f"Saved {len(out)} rows → {out_path}")
    print(f"\nFinal topic distribution:")
    for topic, count in out["topic"].value_counts().items():
        print(f"  {topic:15s}: {count}")
    if "verdict_label" in out.columns:
        print(f"\nVerdict distribution:")
        for v, c in out["verdict_label"].value_counts().items():
            print(f"  {v:20s}: {c}")


if __name__ == "__main__":
    main()
