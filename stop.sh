#!/usr/bin/env bash
# Stop all fakenews Docker containers (data is preserved in ./data/)
# Usage: sudo ./stop.sh
set -e
cd "$(dirname "$0")"
docker compose down
echo "Containers stopped. Data preserved in ./data/"
