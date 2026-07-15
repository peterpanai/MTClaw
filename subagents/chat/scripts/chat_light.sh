#!/bin/bash
set -euo pipefail
INPUT=$(cat)
echo "$INPUT" | python3 "$(dirname "$0")/../chat_engine.py" chat
