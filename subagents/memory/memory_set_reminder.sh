#!/bin/bash
set -euo pipefail

# memory_set_reminder — wrapper for memory_engine.py set_reminder command
# Called by MTClaw Function Router. Reads JSON on stdin, outputs JSON on stdout.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${SCRIPT_DIR}/memory_engine.py"

# Use venv Python if available, otherwise system python3
if [ -f "${SCRIPT_DIR}/.venv/bin/python3" ]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python3"
else
    PYTHON="python3"
fi

error_exit() {
    echo "{\"error\":\"$1\"}"
    exit 1
}

[ ! -f "$ENGINE" ] && error_exit "memory_engine.py not found: $ENGINE"

# Read input JSON from stdin
INPUT=$(cat)

TITLE=$(echo "$INPUT" | jq -r '.title | select(. != null)')
WHEN=$(echo "$INPUT" | jq -r '.when | select(. != null)')
NOTE=$(echo "$INPUT" | jq -r '.note | select(. != null)')

# Validate required parameters
[ -z "$TITLE" ] && error_exit "missing required parameter: title"
[ -z "$WHEN" ] && error_exit "missing required parameter: when"

# Build command
ARGS=("set_reminder" "--title" "$TITLE" "--when" "$WHEN")

if [ -n "$NOTE" ]; then
    ARGS+=("--note" "$NOTE")
fi

# Execute engine
OUTPUT=$("$PYTHON" "$ENGINE" "${ARGS[@]}" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "{\"error\":\"engine failed\",\"exit_code\":$EXIT_CODE,\"detail\":$(echo "$OUTPUT" | jq -Rs .)}"
    exit 1
fi

# Output is already JSON from the engine
echo "$OUTPUT"
