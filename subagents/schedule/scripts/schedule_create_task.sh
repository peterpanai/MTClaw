#!/bin/bash
set -euo pipefail
INPUT=$(cat)
echo "$INPUT" | python3 "$(dirname "$0")/../schedule_engine.py" create_task
