"""FreshQA freshness-gate benchmark.

Validates that the freshness_check node correctly identifies stale cached verdicts
for questions whose answers change at different rates.

FreshQA categories and expected behaviour (stale cache = 365 days old):
  never-changing  — "Who founded Amazon?"           → revalidate=False
  slow-changing   — "Who is the CEO of Google?"     → revalidate=True
  fast-changing   — "What is Tesla's stock price?"  → revalidate=True
  false-premise   — "Who is the King of France?"    → revalidate=True

No Tavily calls are made. Only the freshness classifier (check_freshness) is
exercised. The benchmark measures whether the LLM correctly gates on claim type
and staleness — not whether the downstream answer is correct.

Obtaining FreshQA data:
    Download freshqa.csv from https://github.com/freshllms/freshqa and pass the
    path via --data-path, or use --hf to stream it from HuggingFace.

Usage (local):
    python -m fact_check_agent.src.benchmark.freshqa_benchmark \\
        --data-path /path/to/freshqa.csv \\
        --model gemma4:e2b --stale-days 365 --sample 50

Usage (Langfuse evaluation run):
    # Add to .env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    python -m fact_check_agent.src.benchmark.freshqa_benchmark \\
        --data-path freshqa.csv --langfuse \\
        --dataset-name freshqa-v1 --run-name freshness-gemma4-365d \\
        --sample 50

    # Re-run against existing dataset (skip re-upload):
    python -m fact_check_agent.src.benchmark.freshqa_benchmark \\
        --data-path freshqa.csv --langfuse --run-name freshness-qwen3-365d \\
        --dataset-name freshqa-v1 --no-upload

    # Stream from HuggingFace:
    python -m fact_check_agent.src.benchmark.freshqa_benchmark \\
        --hf --langfuse --run-name freshness-v1
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from langfuse import Langfuse

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parents[4] / "datasets" / "eval_dataset"

# Per-category: expected revalidate given a stale cache
EXPECTED_REVALIDATE: dict[str, bool] = {
    "never-changing": False,
    "slow-changing":  True,
    "fast-changing":  True,
    "false-premise":  True,
}

_QUESTION_ALIASES = ["question", "Question", "claim", "text"]
_CATEGORY_ALIASES = ["category", "Category", "change_frequency", "type"]
_ANSWER_ALIASES   = ["answer",   "Answer",   "ground_truth"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    for name in aliases:
        if name in df.columns:
            return name
    return None


def load_freshqa_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    q_col = _col(df, _QUESTION_ALIASES)
    c_col = _col(df, _CATEGORY_ALIASES)
    if not q_col or not c_col:
        raise ValueError(
            f"Cannot find question/category columns in {path}. "
            f"Available: {list(df.columns)}"
        )
    rename = {q_col: "question", c_col: "category"}
    a_col = _col(df, _ANSWER_ALIASES)
    if a_col:
        rename[a_col] = "answer"
    df = df.rename(columns=rename)
    df["question"] = df["question"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().str.lower().str.replace("_", "-")
    df = df.dropna(subset=["question", "category"])
    df = df[df["question"] != ""]
    return df.reset_index(drop=True)


def load_freshqa_hf() -> pd.DataFrame:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install `datasets` to use --hf: pip install datasets")
    ds = load_dataset("ritheshd30/freshqa", split="train")
    df = ds.to_pandas()
    q_col = _col(df, _QUESTION_ALIASES)
    c_col = _col(df, _CATEGORY_ALIASES)
    if not q_col or not c_col:
        raise ValueError(f"Unexpected HuggingFace schema — columns: {list(df.columns)}")
    rename = {q_col: "question", c_col: "category"}
    a_col = _col(df, _ANSWER_ALIASES)
    if a_col:
        rename[a_col] = "answer"
    df = df.rename(columns=rename)
    df["category"] = df["category"].str.lower().str.replace("_", "-")
    return df.reset_index(drop=True)


def _subsample(df: pd.DataFrame, categories: list[str], sample: int, seed: int) -> pd.DataFrame:
    frames = []
    for cat in categories:
        sub = df[df["category"] == cat]
        n = min(sample, len(sub))
        if n > 0:
            frames.append(sub.sample(n=n, random_state=seed))
    return pd.concat(frames, ignore_index=True) if frames else df


def _stale_timestamp(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


# ── Langfuse helpers ──────────────────────────────────────────────────────────

def _make_langfuse_client() -> "Langfuse":
    """Build Langfuse client from environment variables."""
    from langfuse import Langfuse  # type: ignore
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host       = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    if not public_key or not secret_key:
        raise EnvironmentError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env or environment.\n"
            "  1. Open http://localhost:3000 and create a project\n"
            "  2. Go to Settings → API Keys → Create\n"
            "  3. Add to .env:\n"
            "       LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
            "       LANGFUSE_SECRET_KEY=sk-lf-...\n"
            "       LANGFUSE_HOST=http://localhost:3000"
        )
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def upload_langfuse_dataset(
    df: pd.DataFrame,
    lf: "Langfuse",
    dataset_name: str,
    stale_days: int,
) -> None:
    """Create (or reuse) a Langfuse dataset and upload FreshQA items."""
    try:
        lf.get_dataset(dataset_name)
        print(f"Langfuse dataset '{dataset_name}' already exists — skipping creation")
    except Exception:
        lf.create_dataset(
            name=dataset_name,
            description=f"FreshQA freshness-gate benchmark (stale_days={stale_days})",
            metadata={"stale_days": stale_days, "source": "FreshQA"},
        )
        print(f"Created Langfuse dataset '{dataset_name}'")

    print(f"Uploading {len(df)} items to dataset '{dataset_name}'...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Uploading"):
        category = str(row["category"])
        expected = EXPECTED_REVALIDATE.get(category)
        lf.create_dataset_item(
            dataset_name=dataset_name,
            input={
                "question":    str(row["question"]),
                "category":    category,
                "stale_days":  stale_days,
                "verdict_label":      "supported",
                "verdict_confidence": 0.85,
            },
            expected_output={"revalidate": expected},
            metadata={"answer": str(row.get("answer", ""))},
        )
    lf.flush()
    print(f"Upload complete — {len(df)} items in '{dataset_name}'")


# ── Core benchmark ─────────────────────────────────────────────────────────────

def _run_one(
    question: str,
    category: str,
    last_verified_at: datetime,
    model: str,
) -> dict:
    """Run check_freshness() for one question. Returns a flat result dict."""
    from fact_check_agent.src.tools.freshness_tool import check_freshness
    from fact_check_agent.src.config import settings

    # Route to Ollama for this model regardless of what .env specifies
    settings.llm_provider      = "ollama"
    settings.ollama_llm_model  = model

    expected = EXPECTED_REVALIDATE.get(category)
    try:
        result = check_freshness(
            claim_text         = question,
            verdict_label      = "supported",
            verdict_confidence = 0.85,
            last_verified_at   = last_verified_at,
            api_key            = "",
            model              = model,
        )
        actual       = result["revalidate"]
        llm_category = result.get("claim_category", "unknown")
        reason       = result.get("reason", "")
        error        = None
    except Exception as exc:
        actual       = None
        llm_category = "error"
        reason       = str(exc)
        error        = str(exc)

    correct = (actual == expected) if (expected is not None and actual is not None) else None
    return {
        "question":            question,
        "category":            category,
        "expected_revalidate": expected,
        "actual_revalidate":   actual,
        "correct":             correct,
        "llm_claim_category":  llm_category,
        "reason":              reason,
        "error":               error,
    }


def run_benchmark(
    df: pd.DataFrame,
    model: str,
    stale_days: int,
    categories: list[str],
    sample: Optional[int],
    seed: int,
    lf: Optional["Langfuse"] = None,
    dataset_name: str = "freshqa",
    run_name: str = "freshness-run",
    upload: bool = True,
) -> pd.DataFrame:
    """Run the freshness benchmark, optionally logging each call to Langfuse.

    When `lf` is provided:
      - Uploads FreshQA items as a Langfuse dataset (unless upload=False).
      - Runs the evaluation using dataset items as the source of truth.
      - Each check_freshness() call becomes a Langfuse trace.
      - Scores 'correct' (BOOLEAN) and 'revalidate_decision' (NUMERIC 0/1)
        are attached to every trace.
      - All traces are linked to the dataset run so results appear under
        Langfuse → Datasets → <dataset_name> → Runs → <run_name>.
    """
    df = df[df["category"].isin(categories)].copy()
    if df.empty:
        raise ValueError(
            f"No rows match categories {categories}. "
            f"Available: {df['category'].unique().tolist()}"
        )
    if sample:
        df = _subsample(df, categories, sample, seed)

    n = len(df)
    cat_counts = df["category"].value_counts().to_dict()
    print(f"\nRunning freshness benchmark — {n} datapoints, {stale_days}-day-old simulated cache")
    for cat in sorted(cat_counts):
        exp = EXPECTED_REVALIDATE.get(cat, "?")
        print(f"  {cat:<20s}: {cat_counts[cat]:>4d}  (expected revalidate={exp})")
    print(f"Model: {model}")
    if lf:
        print(f"Langfuse: dataset='{dataset_name}'  run='{run_name}'\n")

    last_verified_at = _stale_timestamp(stale_days)

    # ── Langfuse dataset upload ────────────────────────────────────────────────
    if lf and upload:
        upload_langfuse_dataset(df, lf, dataset_name, stale_days)

    # ── Evaluation loop ────────────────────────────────────────────────────────
    records: list[dict] = []

    if lf:
        dataset = lf.get_dataset(dataset_name)
        items_by_question = {item.input["question"]: item for item in dataset.items}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Freshness checks"):
        question = str(row["question"])
        category = str(row["category"])

        rec = _run_one(question, category, last_verified_at, model)

        if lf:
            # Create a trace for this freshness check
            trace = lf.trace(
                name     = "freshness_check",
                input    = {
                    "question":            question,
                    "category":            category,
                    "stale_days":          stale_days,
                    "verdict_label":       "supported",
                    "verdict_confidence":  0.85,
                },
                output   = {
                    "revalidate":       rec["actual_revalidate"],
                    "claim_category":   rec["llm_claim_category"],
                    "reason":           rec["reason"],
                },
                metadata = {
                    "model":               model,
                    "expected_revalidate": rec["expected_revalidate"],
                    "correct":             rec["correct"],
                    "error":               rec["error"],
                },
                tags     = [category, model, run_name],
            )

            # Log the LLM call as a generation span inside the trace
            trace.generation(
                name   = "check_freshness_llm",
                model  = model,
                input  = question,
                output = rec["reason"],
            )

            # Score: binary correctness
            if rec["correct"] is not None:
                lf.score(
                    trace_id  = trace.id,
                    name      = "correct",
                    value     = 1.0 if rec["correct"] else 0.0,
                    data_type = "BOOLEAN",
                    comment   = f"expected revalidate={rec['expected_revalidate']}",
                )

            # Score: revalidate decision (numeric so it can be averaged per run)
            if rec["actual_revalidate"] is not None:
                lf.score(
                    trace_id  = trace.id,
                    name      = "revalidate_decision",
                    value     = 1.0 if rec["actual_revalidate"] else 0.0,
                    data_type = "NUMERIC",
                )

            # Link trace to dataset run — run_metadata shows as columns in the
            # Langfuse Datasets → Runs comparison table
            item = items_by_question.get(question)
            if item:
                item.link(
                    trace,
                    run_name,
                    run_metadata={
                        "model":       model,
                        "stale_days":  stale_days,
                        "categories":  sorted(df["category"].unique().tolist()),
                        "n_datapoints": len(df),
                    },
                )

            rec["trace_id"] = trace.id

        records.append(rec)

    if lf:
        lf.flush()
        print(f"\nLangfuse run '{run_name}' complete — view at {os.environ.get('LANGFUSE_HOST', 'http://localhost:3000')}")

    return pd.DataFrame(records)


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(results: pd.DataFrame, model: str = "") -> None:
    header = f"FRESHNESS GATE — {model}" if model else "FRESHNESS GATE BENCHMARK — RESULTS"
    print("\n" + "=" * 65)
    print(header)
    print("=" * 65)

    overall_correct = overall_total = 0

    for cat in sorted(results["category"].unique()):
        sub       = results[results["category"] == cat]
        evaluated = sub[sub["correct"].notna()]
        correct   = int(evaluated["correct"].sum())
        total     = len(evaluated)
        expected  = EXPECTED_REVALIDATE.get(cat, "?")
        acc       = correct / total if total > 0 else 0.0

        actual_true  = int((sub["actual_revalidate"] == True).sum())
        actual_false = int((sub["actual_revalidate"] == False).sum())
        errors       = int(sub["error"].notna().sum())

        overall_correct += correct
        overall_total   += total

        print(f"\n[{cat}]  (expected revalidate={expected})")
        print(f"  Accuracy : {correct}/{total} = {acc:.1%}  "
              f"(True:{actual_true}  False:{actual_false}  err:{errors})")

        failures = sub[(sub["correct"] == False) & sub["error"].isna()]
        if not failures.empty:
            print(f"  Failures ({len(failures)} total, showing up to 3):")
            for _, r in failures.head(3).iterrows():
                print(f"    ✗  [{r['llm_claim_category']}] {str(r['question'])[:75]}")
                print(f"       {str(r['reason'])[:110]}")

    overall_acc = overall_correct / overall_total if overall_total else 0.0
    print(f"\n{'─' * 65}")
    print(f"  OVERALL  {overall_correct}/{overall_total} = {overall_acc:.1%}")
    print("=" * 65)


def print_comparison(all_results: dict[str, pd.DataFrame]) -> None:
    """Print a side-by-side accuracy table for all models."""
    categories = sorted({cat for df in all_results.values() for cat in df["category"].unique()})
    models     = list(all_results.keys())
    col_w      = 10

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    # Header
    header = f"{'Category':<22}" + "".join(f"{m[:col_w]:>{col_w}}" for m in models)
    print(header)
    print("─" * len(header))

    for cat in categories:
        row = f"{cat:<22}"
        for model in models:
            df  = all_results[model]
            sub = df[df["category"] == cat]
            ev  = sub[sub["correct"].notna()]
            if len(ev) == 0:
                row += f"{'—':>{col_w}}"
            else:
                acc  = ev["correct"].sum() / len(ev)
                row += f"{acc:>{col_w}.0%}"
        print(row)

    # Overall row
    print("─" * len(header))
    row = f"{'OVERALL':<22}"
    for model in models:
        df  = all_results[model]
        ev  = df[df["correct"].notna()]
        acc = ev["correct"].sum() / len(ev) if len(ev) > 0 else 0.0
        row += f"{acc:>{col_w}.0%}"
    print(row)
    print("=" * 65)


def save_results(results: pd.DataFrame, out_dir: Path, model: str, stale_days: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace(":", "-").replace("/", "-")
    csv = out_dir / f"freshqa_{safe_model}_{stale_days}d_{ts}.csv"
    results.to_csv(csv, index=False)
    print(f"  Saved: {csv}")
    return csv


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FreshQA freshness-gate benchmark — sequential multi-model ablation."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-path", metavar="CSV", help="Path to local freshqa.csv")
    src.add_argument("--hf", action="store_true",  help="Stream from HuggingFace hub")

    parser.add_argument("--models",
                        default="gemma4:e2b",
                        help="Comma-separated models to run sequentially "
                             "(default: gemma4:e2b). "
                             "Example: gemma4:e2b,qwen3:8b,qwen3:1.7b,nemotron-3-nano:4b")
    parser.add_argument("--stale-days",  type=int, default=365,
                        help="Simulated cache age in days (default: 365)")
    parser.add_argument("--categories",
                        default="never-changing,slow-changing,fast-changing,false-premise")
    parser.add_argument("--sample",      type=int, default=None,
                        help="Max questions per category per model (default: all ~150)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no-save",     action="store_true",
                        help="Skip saving per-model results CSV")

    # Langfuse options
    lf_group = parser.add_argument_group("Langfuse (optional)")
    lf_group.add_argument("--langfuse",      action="store_true",
                          help="Log traces and scores to Langfuse")
    lf_group.add_argument("--dataset-name",  default="freshqa",
                          help="Langfuse dataset name (default: freshqa)")
    lf_group.add_argument("--run-name",      default=None,
                          help="Base run name; model name is appended per run. "
                               "Default: freshness-<stale_days>d")
    lf_group.add_argument("--no-upload",     action="store_true",
                          help="Skip uploading items; reuse existing Langfuse dataset")

    args       = parser.parse_args()
    models     = [m.strip() for m in args.models.split(",") if m.strip()]
    categories = [c.strip() for c in args.categories.split(",")]
    base_run   = args.run_name or f"freshness-{args.stale_days}d"

    # Load data once
    if args.hf:
        print("Loading FreshQA from HuggingFace...")
        df = load_freshqa_hf()
    else:
        print(f"Loading FreshQA from {args.data_path}...")
        df = load_freshqa_csv(args.data_path)

    print(f"Loaded {len(df)} questions")
    print(f"Category distribution:\n{df['category'].value_counts().to_string()}")

    # Langfuse client (shared across all model runs)
    lf = None
    if args.langfuse:
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except ImportError:
            pass
        lf = _make_langfuse_client()

    # Sequential model runs
    all_results: dict[str, pd.DataFrame] = {}
    upload_done = False

    for i, model in enumerate(models):
        print(f"\n{'━' * 65}")
        print(f"  Model {i+1}/{len(models)}: {model}")
        print(f"{'━' * 65}")

        run_name = f"{base_run}-{model.replace(':', '-')}"

        results = run_benchmark(
            df           = df,
            model        = model,
            stale_days   = args.stale_days,
            categories   = categories,
            sample       = args.sample,
            seed         = args.seed,
            lf           = lf,
            dataset_name = args.dataset_name,
            run_name     = run_name,
            # Upload dataset on first model run only; subsequent runs reuse it
            upload       = lf is not None and not args.no_upload and not upload_done,
        )

        if lf and not upload_done:
            upload_done = True

        print_report(results, model=model)

        if not args.no_save:
            save_results(results, OUT_DIR, model, args.stale_days)

        all_results[model] = results

    # Comparison table (only meaningful with multiple models)
    if len(models) > 1:
        print_comparison(all_results)
        if not args.no_save:
            combined = pd.concat(
                [df.assign(model=m) for m, df in all_results.items()],
                ignore_index=True,
            )
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = OUT_DIR / f"freshqa_comparison_{args.stale_days}d_{ts}.csv"
            combined.to_csv(out, index=False)
            print(f"\nCombined results saved to: {out}")

    if lf:
        print(f"\nView all runs at: {os.environ.get('LANGFUSE_HOST', 'http://localhost:3000')}"
              f" → Datasets → {args.dataset_name}")


if __name__ == "__main__":
    main()
