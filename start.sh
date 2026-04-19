#!/usr/bin/env bash
# Start Neo4j and ChromaDB containers
# Usage: sudo ./start.sh
set -e
cd "$(dirname "$0")"
docker compose up -d
echo ""
echo "Services:"
echo "  Neo4j Browser  →  http://localhost:7474  (neo4j / fakenews123)"
echo "  ChromaDB API   →  http://localhost:8000"
