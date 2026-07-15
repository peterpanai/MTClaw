#!/bin/bash
set -euo pipefail

# memory_remember — wrapper for memory_engine.py remember command
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

CONTENT=$(echo "$INPUT" | jq -r '.content | select(. != null)')
MTYPE=$(echo "$INPUT" | jq -r '.type | select(. != null)')
TAGS_JSON=$(echo "$INPUT" | jq -c '.tags | select(. != null)')
SOURCE=$(echo "$INPUT" | jq -r '.source | select(. != null)')

# Validate required parameter
[ -z "$CONTENT" ] && error_exit "missing required parameter: content"

# Build command
ARGS=("remember" "--content" "$CONTENT")

if [ -n "$MTYPE" ]; then
    ARGS+=("--type" "$MTYPE")
fi

if [ -n "$SOURCE" ]; then
    ARGS+=("--source" "$SOURCE")
fi

# Tags: convert JSON array to space-separated args
if [ -n "$TAGS_JSON" ] && [ "$TAGS_JSON" != "null" ]; then
    TAG_COUNT=$(echo "$TAGS_JSON" | jq 'length')
    if [ "$TAG_COUNT" -gt 0 ] 2>/dev/null; then
        ARGS+=("--tags")
        while IFS= read -r tag; do
            ARGS+=("$tag")
        done < <(echo "$TAGS_JSON" | jq -r '.[]')
    fi
fi

# Execute engine (stderr goes to stderr, not mixed with stdout JSON)
OUTPUT=$("$PYTHON" "$ENGINE" "${ARGS[@]}" 2>/dev/null)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "{\"error\":\"engine failed\",\"exit_code\":$EXIT_CODE,\"detail\":$(echo "$OUTPUT" | jq -Rs .)}"
    exit 1
fi

# Output is already JSON from the engine
echo "$OUTPUT"
