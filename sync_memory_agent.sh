#!/usr/bin/env bash
# sync_memory_agent.sh — pull the latest memory_agent from the teammate's repo
#
# First run:  clones the repo into memory_agent/
# Subsequent: pulls latest from data_memory_dev branch
#
# Usage:  ./sync_memory_agent.sh

set -euo pipefail

REPO_URL="https://github.com/abdulwasaeee/Agentic-AI-for-Real-Time-News-Analysis-and-Fact-Checking.git"
BRANCH="data_memory_dev"
TARGET="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/memory_agent"

if [[ -d "$TARGET/.git" ]]; then
    echo "── Pulling latest memory_agent (branch: $BRANCH) ──────────────────"
    git -C "$TARGET" fetch origin
    git -C "$TARGET" checkout "$BRANCH"
    git -C "$TARGET" pull origin "$BRANCH"
    echo "✓  memory_agent updated to $(git -C "$TARGET" rev-parse --short HEAD)"
else
    echo "── Cloning memory_agent (branch: $BRANCH) ─────────────────────────"
    # Clone into a temp dir first so a partial clone doesn't leave a broken folder
    TMP=$(mktemp -d)
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$TMP"
    # Preserve any local-only files (e.g. CHANGES.md written by us)
    if [[ -d "$TARGET" ]]; then
        echo "  (existing memory_agent/ found — merging, local-only files kept)"
        rsync -a --ignore-existing "$TARGET/" "$TMP/"
        rm -rf "$TARGET"
    fi
    mv "$TMP" "$TARGET"
    echo "✓  memory_agent cloned at $(git -C "$TARGET" rev-parse --short HEAD)"
fi
