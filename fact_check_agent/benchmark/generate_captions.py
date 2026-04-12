"""Pre-generate VLM image captions for FakeNewsNet records offline.

Run this once before benchmarking to avoid VLM rate-limits during the eval loop.
Results are written back to each BenchmarkRecord.image_caption in-place.

Usage:
    python -m fact_check_agent.benchmark.generate_captions \\
        --dataset-root /path/to/fakenewsnet \\
        --source politifact \\
        --split test
"""
import argparse
import logging
import pickle
from pathlib import Path

from fact_check_agent.src._bootstrap import *  # noqa: F401,F403
from src.preprocessing.caption_generator import CaptionGenerator
from src.config import settings as memory_settings

from fact_check_agent.benchmark.record import BenchmarkRecord, load_fakenewsnet_dataset

logger = logging.getLogger(__name__)


def generate_captions_for_records(
    records: list[BenchmarkRecord],
    cache_path: Path | None = None,
) -> list[BenchmarkRecord]:
    """Generate VLM captions for all records that have image_urls but no caption.

    Optionally saves a cache file (pickle) so you can resume interrupted runs.
    """
    gen = CaptionGenerator(
        api_key=memory_settings.openai_api_key,
        model=memory_settings.llm_model,
    )

    # Load existing cache if provided
    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        logger.info("Loaded caption cache with %d entries from %s", len(cache), cache_path)

    updated = 0
    for record in records:
        if not record.image_urls:
            continue
        if record.image_caption:
            continue

        image_url = record.image_urls[0]

        # Check cache
        if image_url in cache:
            record.image_caption = cache[image_url]
            continue

        try:
            caption = gen.generate_caption(image_url)
            if caption:
                record.image_caption = caption
                cache[image_url] = caption
                updated += 1
        except Exception as e:
            logger.warning("Caption generation failed for %s: %s", record.record_id, e)

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        logger.info("Saved caption cache (%d entries) to %s", len(cache), cache_path)

    logger.info("Generated %d new captions", updated)
    return records


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Pre-generate VLM captions for FakeNewsNet")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source", choices=["politifact", "gossipcop"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache", default="caption_cache.pkl")
    args = parser.parse_args()

    records = load_fakenewsnet_dataset(Path(args.dataset_root), args.source, args.split)
    records = generate_captions_for_records(records, cache_path=Path(args.cache))

    with_captions = sum(1 for r in records if r.image_caption)
    print(f"Done. {with_captions}/{len(records)} records have captions.")


if __name__ == "__main__":
    main()
