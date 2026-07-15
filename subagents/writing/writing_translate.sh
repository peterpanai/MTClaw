#!/bin/bash
set -euo pipefail

# writing_translate - wrapper for writing_engine.py translate command
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

TEXT=$(echo "$INPUT" | jq -r '.text | select(. != null)')
SOURCE_LANG=$(echo "$INPUT" | jq -r '.source_lang | select(. != null)')
TARGET_LANG=$(echo "$INPUT" | jq -r '.target_lang | select(. != null)')
KEEP_FORMATTING=$(echo "$INPUT" | jq -r '.keep_formatting | select(. != null)')

# Validate required parameters
[ -z "$TEXT" ] && error_exit "missing required parameter: text"
[ -z "$TARGET_LANG" ] && error_exit "missing required parameter: target_lang"

# Build JSON payload for engine stdin
PAYLOAD=$(jq -n \
    --arg text "$TEXT" \
    --arg source_lang "${SOURCE_LANG:-auto}" \
    --arg target_lang "$TARGET_LANG" \
    --argjson keep_formatting "${KEEP_FORMATTING:-true}" \
    '{text:$text, source_lang:$source_lang, target_lang:$target_lang, keep_formatting:$keep_formatting}')

# Execute engine
OUTPUT=$(echo "$PAYLOAD" | "$PYTHON" "$ENGINE" writing_translate 2>/dev/null)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "{\"error\":\"engine failed\",\"exit_code\":$EXIT_CODE,\"detail\":$(echo "$OUTPUT" | jq -Rs .)}"
    exit 1
fi

# Output is already JSON from the engine
echo "$OUTPUT"
