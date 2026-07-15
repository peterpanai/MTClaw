#!/bin/bash
set -euo pipefail

# writing_generate - wrapper for writing_engine.py generate command
# Called by MTClaw Function Router. Reads JSON on stdin, outputs JSON on stdout.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${SCRIPT_DIR}/writing_engine.py"

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

[ ! -f "$ENGINE" ] && error_exit "writing_engine.py not found: $ENGINE"

# Read input JSON from stdin
INPUT=$(cat)

DOC_TYPE=$(echo "$INPUT" | jq -r '.doc_type | select(. != null)')
TOPIC=$(echo "$INPUT" | jq -r '.topic | select(. != null)')
KEY_POINTS_JSON=$(echo "$INPUT" | jq -c '.key_points | select(. != null)')
STYLE=$(echo "$INPUT" | jq -r '.style | select(. != null)')
LENGTH=$(echo "$INPUT" | jq -r '.length | select(. != null)')

# Validate required parameters
[ -z "$DOC_TYPE" ] && error_exit "missing required parameter: doc_type"
[ -z "$TOPIC" ] && error_exit "missing required parameter: topic"

# Build JSON payload for engine stdin
PAYLOAD=$(jq -n \
    --arg doc_type "$DOC_TYPE" \
    --arg topic "$TOPIC" \
    --argjson key_points "${KEY_POINTS_JSON:-[]}" \
    --arg style "${STYLE:-formal}" \
    --arg length "${LENGTH:-medium}" \
    '{doc_type:$doc_type, topic:$topic, key_points:$key_points, style:$style, length:$length}')

# Execute engine
OUTPUT=$(echo "$PAYLOAD" | "$PYTHON" "$ENGINE" writing_generate 2>/dev/null)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "{\"error\":\"engine failed\",\"exit_code\":$EXIT_CODE,\"detail\":$(echo "$OUTPUT" | jq -Rs .)}"
    exit 1
fi

# Output is already JSON from the engine
echo "$OUTPUT"
