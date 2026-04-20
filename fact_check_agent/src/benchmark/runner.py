"""Factify2 benchmark runner for the Fact-Check Agent.

Evaluates the pipeline against the Factify2 multi-modal fact-verification dataset.
Uses Option A (document injection) to bypass Tavily — no API keys or network calls.
All inference runs locally via Ollama.

Usage:
    python -m fact_check_agent.src.benchmark.runner [OPTIONS]

    --split      val | train | test          (default: val)
    --limit      max records to evaluate     (default: 200)
    --out        output CSV path             (default: results/benchmark_<timestamp>.csv)
    --no-image   disable image_url field     (default: images enabled)

SOTA flags — toggle in .env before running:
    USE_SIGLIP=true/false              SigLIP image-text similarity (local, no Ollama needed)
    USE_RETRIEVAL_GATE=true/false      Adaptive retrieval gate (S2)
    USE_CLAIM_DECOMPOSITION=true/false Claim decomposition (S3)
    USE_DEBATE=true/false              Multi-agent debate (S4)
    USE_FRESHNESS_REACT=true/false     Freshness ReAct agent (S6)
    LLM_PROVIDER=ollama                Must be 'ollama' for local benchmark
    OLLAMA_LLM_MODEL=gemma4:e2b        Ollama model (or gemma4:12b for better accuracy)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)  # suppress property-key warnings on empty graph
logging.getLogger("langfuse").setLevel(logging.ERROR)             # suppress tracing errors when server is offline
logger = logging.getLogger(__name__)

# ── Label mapping ─────────────────────────────────────────────────────────────

VERDICT_MAP = {
    "Support_Multimodal":      "supported",
    "Support_Text":            "supported",
    "Insufficient_Multimodal": "misleading",
    "Insufficient_Text":       "misleading",
    "Refute":                  "refuted",
}

DATASET_ROOT = (
    Path(__file__).resolve().parents[3] / "datasets" / "Factify2" / "Factify 2"
)
SPLIT_PATHS = {
    "train": DATASET_ROOT / "factify2_train" / "factify2" / "train.csv",
    "val":   DATASET_ROOT / "factify2_train" / "factify2" / "val.csv",
    "test":  DATASET_ROOT / "factify2_test"  / "test.csv",
}


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_factify2(split: str, limit: Optional[int] = None) -> pd.DataFrame:
    path = SPLIT_PATHS[split]
    if not path.exists():
        raise FileNotFoundError(f"Factify2 {split} split not found at {path}")
    df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["claim", "document"])
    df = df[df["claim"].str.strip() != ""]
    df = df[df["document"].str.strip() != ""]
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


# ── Input builder ─────────────────────────────────────────────────────────────

def build_fact_check_input(row: pd.Series, include_image: bool = True):
    """Convert one Factify2 row to a FactCheckInput.

    Option A: inject document text as prefetched_chunks → skips live_search.
    image_url is populated from claim_image when present and include_image=True.
    """
    from fact_check_agent.src.models.schemas import FactCheckInput
    from fact_check_agent.src.id_utils import make_id

    row_idx = str(row.get("Unnamed: 0", row.name))
    claim_text = str(row["claim"]).strip()
    document   = str(row.get("document", "")).strip()
    image_url  = str(row.get("claim_image", "")).strip() or None

    # Truncate document to avoid token limit issues
    evidence_chunk = f"[REFERENCE DOCUMENT]\n{document[:3000]}"

    # OCR text as additional context when present
    claim_ocr = str(row.get("Claim OCR", "")).strip()
    if claim_ocr and claim_ocr not in ("nan", " ", ""):
        evidence_chunk += f"\n\n[CLAIM IMAGE OCR]\n{claim_ocr[:500]}"

    doc_ocr = str(row.get("Document OCR", "")).strip()
    if doc_ocr and doc_ocr not in ("nan", " ", ""):
        evidence_chunk += f"\n\n[DOCUMENT IMAGE OCR]\n{doc_ocr[:500]}"

    # Derive source_url from claim_image domain or use placeholder
    source_url = "https://factify2.benchmark/unknown"
    if image_url and image_url.startswith("http"):
        from urllib.parse import urlparse
        parsed = urlparse(image_url)
        source_url = f"{parsed.scheme}://{parsed.netloc}/"

    return FactCheckInput(
        claim_id     = make_id(f"factify2_{row_idx}_"),
        claim_text   = claim_text,
        entities     = [],
        source_url   = source_url,
        article_id   = f"factify2_{row_idx}",
        image_url    = image_url if include_image else None,
        timestamp    = datetime.now(timezone.utc),
        prefetched_chunks = [evidence_chunk],
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(records: list[dict]) -> dict:
    """Compute accuracy, per-class F1, and confusion matrix from result records."""
    labels = ["supported", "refuted", "misleading"]
    y_true = [r["true_verdict"] for r in records if r["pred_verdict"] is not None]
    y_pred = [r["pred_verdict"] for r in records if r["pred_verdict"] is not None]
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "n": 0}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    # Per-class precision / recall / F1
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                            "support": sum(1 for t in y_true if t == label)}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels)

    # Confusion matrix (rows=true, cols=pred)
    conf_matrix = {}
    for t in labels:
        conf_matrix[t] = {p: sum(1 for yt, yp in zip(y_true, y_pred) if yt == t and yp == p)
                         for p in labels}

    # Error analysis
    n_errors = n - correct
    cross_modal_flagged = sum(1 for r in records if r.get("cross_modal_flag"))
    mean_confidence = sum(r.get("confidence_score", 0) for r in records if r["pred_verdict"]) / n

    return {
        "n": n,
        "n_errors": n_errors,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": conf_matrix,
        "cross_modal_flagged": cross_modal_flagged,
        "cross_modal_flag_rate": round(cross_modal_flagged / n, 4),
        "mean_confidence": round(mean_confidence, 2),
    }


def print_metrics(metrics: dict, settings_snapshot: dict) -> None:
    print("\n" + "=" * 60)
    print("FACTIFY2 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  n={metrics['n']}  accuracy={metrics['accuracy']:.3f}  macro_F1={metrics['macro_f1']:.3f}")
    print(f"  mean_confidence={metrics['mean_confidence']:.1f}  cross_modal_flagged={metrics['cross_modal_flagged']} ({metrics['cross_modal_flag_rate']:.1%})")
    print()
    print(f"  {'Label':<18} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print(f"  {'-'*43}")
    for label, v in metrics["per_class"].items():
        print(f"  {label:<18} {v['precision']:>6.3f} {v['recall']:>6.3f} {v['f1']:>6.3f} {v['support']:>5}")
    print()
    print("  Confusion matrix (rows=true, cols=pred):")
    labels = list(metrics["confusion_matrix"].keys())
    header = f"  {'':18} " + " ".join(f"{l[:7]:>9}" for l in labels)
    print(header)
    for true_label, row in metrics["confusion_matrix"].items():
        vals = " ".join(f"{row[p]:>9}" for p in labels)
        print(f"  {true_label:<18} {vals}")
    print()
    print("  SOTA flags active:")
    for k, v in settings_snapshot.items():
        if v:
            print(f"    {k} = {v}")
    print("=" * 60)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_benchmark(
    split: str = "val",
    limit: Optional[int] = 200,
    output_path: Optional[str] = None,
    include_image: bool = True,
) -> dict:
    """Run the benchmark and return metrics dict.

    All inference is local (Ollama). Document is injected directly as evidence
    (Option A) so Tavily is never called.
    """
    # Bootstrap path resolution for memory_agent imports
    _root = Path(__file__).resolve().parents[3]
    for p in [str(_root / "memory_agent"), str(_root / "fact_check_agent")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Must set env before importing settings
    for key, default in [
        ("LLM_PROVIDER",     "ollama"),
        ("EMBEDDING_PROVIDER","ollama"),
        ("OPENAI_API_KEY",    "unused"),
        ("NEO4J_URI",         "bolt://localhost:7687"),
        ("NEO4J_PASSWORD",    "fakenews123"),
        ("CHROMA_HOST",       "localhost"),
    ]:
        os.environ.setdefault(key, default)

    from fact_check_agent.src.config import settings
    from fact_check_agent.src.graph.graph import build_graph
    from fact_check_agent.src.memory_client import get_memory

    settings_snapshot = {
        "llm_provider":            settings.llm_provider,
        "ollama_llm_model":        settings.ollama_llm_model,
        "use_siglip":              settings.use_siglip,
        "use_retrieval_gate":      settings.use_retrieval_gate,
        "use_claim_decomposition": settings.use_claim_decomposition,
        "use_debate":              settings.use_debate,
        "use_freshness_react":     settings.use_freshness_react,
        "include_image":           include_image,
        "split":                   split,
        "limit":                   limit,
    }

    print(f"\nLoading Factify2 {split} split (limit={limit})...")
    df = load_factify2(split, limit)
    print(f"  {len(df)} records loaded")

    print("Connecting to MemoryAgent...")
    try:
        memory = get_memory()
    except Exception as e:
        print(f"  WARNING: MemoryAgent unavailable ({e}). Running without memory (no cache path).")
        from unittest.mock import MagicMock
        memory = MagicMock()
        memory.search_similar_claims.return_value = {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}
        memory.get_entity_context.return_value = []
        memory.get_entity_ids_for_claims.return_value = []
        memory.get_graph_claims_for_entities.return_value = []
        memory.get_verdict_by_claim.return_value = {"ids": [], "metadatas": []}
        memory.add_verdict.return_value = None
        memory.query_source_credibility.return_value = {"distances": [[]], "metadatas": [[]]}
        memory.add_source_credibility_point.return_value = None

    graph = build_graph(memory)

    results: list[dict] = []
    n_errors = 0
    start_time = time.time()

    print(f"Running benchmark ({len(df)} records)...\n")

    for i, row in df.iterrows():
        true_label_raw = str(row.get("Category", "")).strip()
        true_verdict   = VERDICT_MAP.get(true_label_raw, "misleading")

        try:
            fact_input = build_fact_check_input(row, include_image=include_image)

            state = graph.invoke({"input": fact_input})
            output = state.get("output")

            pred_verdict    = output.verdict          if output else None
            confidence      = output.confidence_score if output else None
            cross_modal_flag= output.cross_modal_flag if output else False
            cross_modal_exp = output.cross_modal_explanation if output else None
            reasoning       = output.reasoning        if output else ""
            bias_score      = output.bias_score       if output else None

        except Exception as e:
            logger.error("Record %d failed: %s", i, e)
            n_errors += 1
            pred_verdict = None
            confidence = None
            cross_modal_flag = False
            cross_modal_exp = None
            reasoning = f"ERROR: {e}"
            bias_score = None

        record = {
            "row_idx":            str(row.get("Unnamed: 0", i)),
            "claim":              str(row["claim"])[:200],
            "true_category":      true_label_raw,
            "true_verdict":       true_verdict,
            "pred_verdict":       pred_verdict,
            "confidence_score":   confidence,
            "bias_score":         bias_score,
            "cross_modal_flag":   cross_modal_flag,
            "cross_modal_explanation": cross_modal_exp,
            "correct":            pred_verdict == true_verdict if pred_verdict else False,
            "reasoning":          reasoning[:300],
            "has_claim_image":    bool(str(row.get("claim_image", "")).strip() not in ("", "nan")),
        }
        results.append(record)

        # Progress reporting
        done = i + 1 if isinstance(i, int) else len(results)
        if done % 10 == 0 or done == len(df):
            elapsed = time.time() - start_time
            acc_so_far = sum(1 for r in results if r["correct"]) / len(results)
            print(f"  [{done:>4}/{len(df)}] acc={acc_so_far:.3f}  errors={n_errors}  elapsed={elapsed:.0f}s")

    total_time = time.time() - start_time
    metrics = compute_metrics(results)
    metrics["total_time_s"] = round(total_time, 1)
    metrics["n_pipeline_errors"] = n_errors
    metrics["settings"] = settings_snapshot

    print_metrics(metrics, settings_snapshot)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        out_dir = _root / "results"
        out_dir.mkdir(exist_ok=True)
        output_path = str(out_dir / f"benchmark_{split}_{timestamp}.csv")

    metrics_path = output_path.replace(".csv", "_metrics.json")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print(f"Metrics saved to: {metrics_path}")

    return metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Factify2 benchmark for the Fact-Check Agent")
    parser.add_argument("--split",    default="val", choices=["val", "train", "test"],
                        help="Dataset split to evaluate (default: val)")
    parser.add_argument("--limit",    type=int, default=200,
                        help="Max records to evaluate (default: 200; set 0 for full split)")
    parser.add_argument("--out",      default=None,
                        help="Output CSV path (default: results/benchmark_<timestamp>.csv)")
    parser.add_argument("--no-image", action="store_true",
                        help="Disable image_url field (text-only mode)")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    run_benchmark(
        split        = args.split,
        limit        = limit,
        output_path  = args.out,
        include_image= not args.no_image,
    )


if __name__ == "__main__":
    main()
