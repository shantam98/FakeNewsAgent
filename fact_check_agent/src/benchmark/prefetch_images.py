"""Pre-fetch Factify2 dataset images to local disk and curate the dataset.

Downloads all unique image URLs from claim_image and document_image columns
across requested splits, saving files to:
    datasets/Factify2/images/<md5_of_url>.<ext>

Produces:
    datasets/Factify2/url_to_local.json   — url → local path (all successes)
    datasets/Factify2/Factify 2/factify2_train/factify2/val_curated.csv
    datasets/Factify2/Factify 2/factify2_test/test_curated.csv
    (etc.)  — rows where claim_image downloaded successfully, with local paths

The curated CSVs can be used directly as benchmark input with --split val_curated.

Usage:
    python -m fact_check_agent.src.benchmark.prefetch_images [OPTIONS]

    --splits     val,test         Comma-separated splits to process (default: val,test)
    --workers    N                Parallel download threads (default: 16)
    --timeout    N                Per-request timeout in seconds (default: 10)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_ROOT = (
    Path(__file__).resolve().parents[3] / "datasets" / "Factify2" / "Factify 2"
)
IMAGES_DIR = DATASET_ROOT.parent / "images"
MAPPING_PATH = DATASET_ROOT.parent / "url_to_local.json"

SPLIT_PATHS = {
    "train": DATASET_ROOT / "factify2_train" / "factify2" / "train.csv",
    "val":   DATASET_ROOT / "factify2_train" / "factify2" / "val.csv",
    "test":  DATASET_ROOT / "factify2_test"  / "test.csv",
}

# Curated output lives alongside the originals
CURATED_PATHS = {
    "train": DATASET_ROOT / "factify2_train" / "factify2" / "train_curated.csv",
    "val":   DATASET_ROOT / "factify2_train" / "factify2" / "val_curated.csv",
    "test":  DATASET_ROOT / "factify2_test"  / "test_curated.csv",
}

IMAGE_COLUMNS = ["claim_image", "document_image"]


def _url_to_filename(url: str) -> str:
    """Stable filename: md5 of URL + original extension."""
    ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png", "gif", "webp", "bmp"):
        ext = "jpg"
    return f"{hashlib.md5(url.encode()).hexdigest()}.{ext}"


def _download(url: str, dest: Path, timeout: int) -> Optional[str]:
    """Download url → dest. Returns local path on success, None on failure."""
    if dest.exists():
        return str(dest)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            dest.write_bytes(r.read())
        return str(dest)
    except Exception as e:
        logger.debug("Failed %s: %s", url, e)
        return None


def collect_urls(splits: list[str]) -> list[str]:
    """Deduplicated list of all image URLs across requested splits."""
    seen: set[str] = set()
    for split in splits:
        path = SPLIT_PATHS.get(split)
        if not path or not path.exists():
            logger.warning("Split '%s' not found — skipping", split)
            continue
        df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
        for col in IMAGE_COLUMNS:
            if col not in df.columns:
                continue
            for val in df[col].dropna():
                url = str(val).strip()
                if url and url != "nan" and url.startswith("http"):
                    seen.add(url)
    return sorted(seen)


def prefetch(splits: list[str], workers: int, timeout: int) -> dict[str, str]:
    """Download all images. Returns mapping url → local_path for successes."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    existing: dict[str, str] = {}
    if MAPPING_PATH.exists():
        with MAPPING_PATH.open() as f:
            existing = json.load(f)

    urls = collect_urls(splits)
    pending = [u for u in urls if u not in existing]
    print(f"\nTotal unique URLs : {len(urls)}")
    print(f"Already cached    : {len(existing)}")
    print(f"To download       : {len(pending)}")

    mapping: dict[str, str] = dict(existing)
    ok = failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download, url, IMAGES_DIR / _url_to_filename(url), timeout): url
            for url in pending
        }
        with tqdm(total=len(pending), desc="Downloading", unit="img") as bar:
            for future in as_completed(futures):
                url = futures[future]
                result = future.result()
                if result:
                    mapping[url] = result
                    ok += 1
                else:
                    failed += 1
                bar.update(1)
                bar.set_postfix(ok=ok, fail=failed)

    with MAPPING_PATH.open("w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nDownload complete — ok={ok}  failed={failed}  total_cached={len(mapping)}")
    print(f"Mapping : {MAPPING_PATH}")
    return mapping


def curate(splits: list[str], mapping: dict[str, str]) -> None:
    """Write curated CSVs containing only rows where claim_image was fetched.

    Both claim_image and document_image columns are replaced with local paths
    where available. Rows missing a local claim_image are dropped.
    """
    for split in splits:
        src = SPLIT_PATHS.get(split)
        dst = CURATED_PATHS.get(split)
        if not src or not src.exists():
            continue

        df = pd.read_csv(src, sep="\t", engine="python", on_bad_lines="skip")
        df = df.dropna(subset=["claim", "document"])
        df = df[df["claim"].str.strip() != ""]
        df = df[df["document"].str.strip() != ""]

        # Replace URLs with local paths for both image columns
        for col in IMAGE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda u: mapping.get(str(u).strip(), "") if pd.notna(u) else ""
                )

        # Keep only rows where claim_image was successfully fetched
        before = len(df)
        df = df[df["claim_image"].apply(lambda p: bool(p) and Path(p).exists())]
        after = len(df)

        df.to_csv(dst, sep="\t", index=False)
        print(f"\n[{split}] {before} rows → {after} curated ({before - after} dropped)")
        print(f"  Saved: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fetch Factify2 images and curate dataset.")
    parser.add_argument("--splits", default="val,test",
                        help="Comma-separated splits (default: val,test)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel download threads (default: 16)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Per-request timeout in seconds (default: 10)")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    mapping = prefetch(splits, args.workers, args.timeout)
    print("\nCurating dataset...")
    curate(splits, mapping)


if __name__ == "__main__":
    main()
