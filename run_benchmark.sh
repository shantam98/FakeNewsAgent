#!/usr/bin/env bash
# =============================================================================
# Factify2 Benchmark Runner
#
# Usage:
#   ./run_benchmark.sh                        # baseline, val split, 200 records
#   ./run_benchmark.sh --split val --limit 500
#   ./run_benchmark.sh --split train --limit 0   # full split
#   ./run_benchmark.sh --no-image               # text-only mode
#
# SOTA flags are toggled via .env before running:
#   USE_SIGLIP=true              S5: SigLIP cross-modal (local, no Ollama needed)
#   USE_RETRIEVAL_GATE=true      S2: skip Tavily when memory is sufficient
#   USE_CLAIM_DECOMPOSITION=true S3: split compound claims before retrieval
#   USE_DEBATE=true              S4: advocate/arbiter debate for low-confidence
#   USE_FRESHNESS_REACT=true     S6: ReAct freshness check
#
# Output:
#   results/benchmark_<split>_<timestamp>.csv
#   results/benchmark_<split>_<timestamp>_metrics.json
#   logs/benchmark_<timestamp>.log
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p results logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/benchmark_${TIMESTAMP}.log"

# ── Environment ───────────────────────────────────────────────────────────────
# Load .env from fact_check_agent if present
ENV_FILE="fact_check_agent/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

export PYTHONPATH="${SCRIPT_DIR}/memory_agent:${SCRIPT_DIR}/fact_check_agent"

# ── Print config header ───────────────────────────────────────────────────────
print_header() {
    echo "============================================================"
    echo "  Factify2 Benchmark  —  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    echo "  LLM provider  : ${LLM_PROVIDER:-openai}"
    echo "  LLM model     : ${OLLAMA_LLM_MODEL:-${LLM_MODEL:-gpt-4o}}"
    echo "  Embedding     : ${EMBEDDING_PROVIDER:-openai} / ${OLLAMA_EMBEDDING_MODEL:-${EMBEDDING_MODEL:-text-embedding-3-small}}"
    echo ""
    echo "  SOTA flags:"
    echo "    USE_SIGLIP              = ${USE_SIGLIP:-false}"
    echo "    USE_RETRIEVAL_GATE      = ${USE_RETRIEVAL_GATE:-false}"
    echo "    USE_CLAIM_DECOMPOSITION = ${USE_CLAIM_DECOMPOSITION:-false}"
    echo "    USE_DEBATE              = ${USE_DEBATE:-false}"
    echo "    USE_FRESHNESS_REACT     = ${USE_FRESHNESS_REACT:-false}"
    echo ""
    echo "  Log file: $LOG_FILE"
    echo "============================================================"
}

# ── Run ───────────────────────────────────────────────────────────────────────
print_header 2>&1 | tee "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Starting benchmark at $(date '+%H:%M:%S')..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

START_EPOCH=$(date +%s)

.venv/bin/python \
    -m fact_check_agent.src.benchmark.runner \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

END_EPOCH=$(date +%s)
ELAPSED=$(( END_EPOCH - START_EPOCH ))

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  Benchmark completed successfully in ${ELAPSED}s" | tee -a "$LOG_FILE"
else
    echo "  Benchmark FAILED (exit code $EXIT_CODE) after ${ELAPSED}s" | tee -a "$LOG_FILE"
fi
echo "  Full log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
