"""End-to-end benchmark runner for LIAR and FakeNewsNet.

Usage:
    # LIAR test split
    python -m fact_check_agent.benchmark.run_eval \\
        --dataset liar \\
        --liar-path /path/to/liar/test.tsv \\
        --split test

    # FakeNewsNet PolitiFact
    python -m fact_check_agent.benchmark.run_eval \\
        --dataset fakenewsnet \\
        --fnn-root /path/to/fakenewsnet \\
        --source politifact \\
        --split test

    # Seed memory first then eval (optional — improves cache-hit path)
    python -m fact_check_agent.benchmark.run_eval \\
        --dataset liar --liar-path .../train.tsv --split train --seed-only
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from sklearn.metrics import classification_report, f1_score

from fact_check_agent.src.memory_client import close_memory, get_memory
from fact_check_agent.src.graph.graph import build_graph
from fact_check_agent.src.models.schemas import FactCheckOutput
from fact_check_agent.benchmark.record import (
    BenchmarkRecord,
    load_fakenewsnet_dataset,
    load_liar_dataset,
)

logger = logging.getLogger(__name__)


def seed_memory(records: list[BenchmarkRecord]) -> None:
    """Optionally seed MemoryAgent with benchmark records before eval.

    Use the training split to populate the vector/graph stores so the
    cache-hit path is exercised during test evaluation.
    """
    memory = get_memory()
    seeded = skipped = 0
    for record in records:
        output = record.to_preprocessing_output()
        was_new = memory.ingest_preprocessed(output)
        if was_new:
            seeded += 1
        else:
            skipped += 1
    logger.info("Seeded %d records (%d duplicates skipped)", seeded, skipped)


def run_eval(
    records: list[BenchmarkRecord],
    dataset_name: str,
    output_path: Optional[Path] = None,
) -> dict:
    """Run the fact-check graph on all records and compute metrics.

    Returns a results dict with macro_f1, y_true, y_pred, and per-record outputs.
    """
    memory = get_memory()
    graph  = build_graph(memory)

    y_true: list[int] = []
    y_pred: list[int] = []
    result_rows: list[dict] = []

    for i, record in enumerate(records):
        fact_check_input = record.to_fact_check_input()
        try:
            state  = graph.invoke({"input": fact_check_input})
            output: Optional[FactCheckOutput] = state.get("output")
        except Exception as e:
            logger.error("Graph failed for record %s: %s", record.record_id, e)
            output = None

        if output is None:
            logger.warning("No output for record %s — treating as misleading", record.record_id)
            pred_binary = 1
            pred_label  = "misleading"
            confidence  = 0
        else:
            pred_binary = 1 if output.verdict in ("refuted", "misleading") else 0
            pred_label  = output.verdict
            confidence  = output.confidence_score

        y_true.append(record.ground_truth_binary)
        y_pred.append(pred_binary)
        result_rows.append({
            "record_id":           record.record_id,
            "claim_text":          record.claim_text[:120],
            "ground_truth_label":  record.ground_truth_label,
            "ground_truth_binary": record.ground_truth_binary,
            "predicted_verdict":   pred_label,
            "predicted_binary":    pred_binary,
            "confidence_score":    confidence,
            "correct":             record.ground_truth_binary == pred_binary,
        })

        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d", i + 1, len(records))

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report   = classification_report(
        y_true, y_pred, target_names=["real", "fake"], zero_division=0
    )

    print(f"\n{'='*60}")
    print(f"  {dataset_name} — Evaluation Results")
    print(f"{'='*60}")
    print(report)
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"{'='*60}\n")

    results = {
        "dataset":   dataset_name,
        "n_records": len(records),
        "macro_f1":  macro_f1,
        "y_true":    y_true,
        "y_pred":    y_pred,
        "rows":      result_rows,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(
                {k: v for k, v in results.items() if k != "rows"} | {"rows": result_rows},
                f, indent=2,
            )
        logger.info("Results saved to %s", output_path)

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run fact-check agent benchmark eval")
    parser.add_argument("--dataset", choices=["liar", "fakenewsnet"], required=True)
    parser.add_argument("--liar-path",  help="Path to LIAR TSV file (train/valid/test)")
    parser.add_argument("--fnn-root",   help="Path to FakeNewsNet dataset root directory")
    parser.add_argument("--source",     choices=["politifact", "gossipcop"], default="politifact")
    parser.add_argument("--split",      default="test")
    parser.add_argument("--seed-only",  action="store_true", help="Seed memory and exit")
    parser.add_argument("--seed-train", help="LIAR train.tsv to seed memory before eval")
    parser.add_argument("--output",     help="Save JSON results to this path")
    args = parser.parse_args()

    # Load records
    if args.dataset == "liar":
        if not args.liar_path:
            parser.error("--liar-path required for liar dataset")
        records = load_liar_dataset(args.liar_path, split=args.split)
        name    = f"LIAR ({args.split})"
    else:
        if not args.fnn_root:
            parser.error("--fnn-root required for fakenewsnet dataset")
        records = load_fakenewsnet_dataset(Path(args.fnn_root), args.source, args.split)
        name    = f"FakeNewsNet-{args.source} ({args.split})"

    # Optional: seed memory with train split first
    if args.seed_train:
        train_records = load_liar_dataset(args.seed_train, split="train")
        seed_memory(train_records)

    if args.seed_only:
        seed_memory(records)
        close_memory()
        return

    output_path = Path(args.output) if args.output else None
    run_eval(records, name, output_path=output_path)
    close_memory()


if __name__ == "__main__":
    main()
