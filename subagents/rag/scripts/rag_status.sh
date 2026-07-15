#!/bin/bash
set -euo pipefail

# RAG status wrapper - reads JSON from stdin, calls rag_engine.py, outputs JSON to stdout
# Called by MTClaw Function Router: execute_tool("rag_status", '{}')

INPUT=$(cat)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROMETHEUS_DIR="${HOME}/.prometheus"
DATA_DIR="${RAG_DATA_DIR:-${PROMETHEUS_DIR}/data}"

if [ -f "${PROMETHEUS_DIR}/python_tools/rag_engine.py" ]; then
    ENGINE="${PROMETHEUS_DIR}/python_tools/rag_engine.py"
elif [ -f "${SCRIPT_DIR}/../rag_engine.py" ]; then
    ENGINE="${SCRIPT_DIR}/../rag_engine.py"
else
    echo '{"error":"rag_engine.py not found"}'
    exit 1
fi

echo "$INPUT" | python3 "$ENGINE" rag_status --data-dir "$DATA_DIR"
