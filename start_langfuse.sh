#!/usr/bin/env bash
# Start Langfuse and its Postgres DB containers
# Usage: sudo ./start_langfuse.sh
set -e
cd "$(dirname "$0")"
docker compose up -d langfuse-db langfuse
echo ""
echo "Services:"
echo "  Langfuse UI  →  http://localhost:3000"
echo ""
echo "First run: create an account at http://localhost:3000, then add keys to .env:"
echo "  LANGFUSE_PUBLIC_KEY=pk-lf-..."
echo "  LANGFUSE_SECRET_KEY=sk-lf-..."
echo "  LANGFUSE_HOST=http://localhost:3000"
