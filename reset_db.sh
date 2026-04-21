#!/usr/bin/env bash
# reset_db.sh — wipe ChromaDB and Neo4j back to a clean slate
# Usage:
#   ./reset_db.sh           # prompt for confirmation
#   ./reset_db.sh --yes     # skip prompt (CI / scripted use)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/fact_check_agent/.env"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Fall back to system python if venv not found
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

# ── Load .env ──────────────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CHROMA_HOST="${CHROMA_HOST:-localhost}"
CHROMA_PORT="${CHROMA_PORT:-8000}"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-fakenews123}"

# ── Confirmation ───────────────────────────────────────────────────────────
if [[ "${1:-}" != "--yes" ]]; then
    echo "⚠️  This will DELETE ALL DATA in:"
    echo "   ChromaDB  → http://${CHROMA_HOST}:${CHROMA_PORT}"
    echo "   Neo4j     → ${NEO4J_URI}"
    read -r -p "Type 'yes' to continue: " CONFIRM
    if [[ "$CONFIRM" != "yes" ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ── Reset ChromaDB ─────────────────────────────────────────────────────────
echo ""
echo "── ChromaDB reset ────────────────────────────────────────────────────"
"$PYTHON" - <<PYEOF
import chromadb, sys

client = chromadb.HttpClient(host="${CHROMA_HOST}", port=${CHROMA_PORT})
collections = ["claims", "articles", "verdicts", "image_captions", "source_credibility"]
for name in collections:
    try:
        client.delete_collection(name)
        print(f"  deleted  {name}")
    except Exception as e:
        print(f"  skipped  {name}  ({e})")

# Recreate empty collections so the agent can start immediately
for name in collections:
    client.get_or_create_collection(name)
    print(f"  created  {name}")

print("ChromaDB: done")
PYEOF

# ── Reset Neo4j ────────────────────────────────────────────────────────────
echo ""
echo "── Neo4j reset ───────────────────────────────────────────────────────"
"$PYTHON" - <<PYEOF
from neo4j import GraphDatabase

driver = GraphDatabase.driver("${NEO4J_URI}", auth=("${NEO4J_USER}", "${NEO4J_PASSWORD}"))
with driver.session() as session:
    result = session.run("MATCH (n) DETACH DELETE n")
    summary = result.consume()
    print(f"  deleted  {summary.counters.nodes_deleted} nodes, "
          f"{summary.counters.relationships_deleted} relationships")
driver.close()
print("Neo4j: done")
PYEOF

echo ""
echo "✓  Both databases are clean. Re-run init_schema() on next agent start."
