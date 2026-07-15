#!/usr/bin/env python3
"""
Aggregate all subagent functions.jsonl files into a single combined file.

MTClaw Function Router's load_tools() reads a SINGLE functions.jsonl file.
This script scans all subagents/*/functions.jsonl, deduplicates by tool name
(first occurrence wins), and outputs a sorted combined file.

Usage:
  python3 aggregate_functions.py [--output <path>] [--subagents-dir <path>]

Default output: stdout
Default subagents dir: ./subagents/ (relative to this script's location)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def aggregate(subagents_dir: Path) -> tuple[list[dict], list[str]]:
    """Scan all subagents/*/functions.jsonl and aggregate into one list.

    Returns:
        (tools, warnings) - list of tool dicts and list of warning messages
    """
    tools: list[dict] = []
    seen_names: set[str] = set()
    warnings: list[str] = []

    if not subagents_dir.exists():
        warnings.append(f"subagents directory not found: {subagents_dir}")
        return tools, warnings

    # Sort subagent dirs for deterministic ordering
    subagent_dirs = sorted(
        d for d in subagents_dir.iterdir()
        if d.is_dir() and (d / "functions.jsonl").exists()
    )

    for subagent_dir in subagent_dirs:
        functions_file = subagent_dir / "functions.jsonl"
        try:
            with functions_file.open("r", encoding="utf-8") as f:
                for line_num, raw_line in enumerate(f, 1):
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        tool = json.loads(line)
                    except json.JSONDecodeError as e:
                        warnings.append(
                            f"{functions_file}:{line_num}: JSON parse error: {e}"
                        )
                        continue

                    name = tool.get("name", "")
                    if not name:
                        warnings.append(
                            f"{functions_file}:{line_num}: tool missing 'name' field"
                        )
                        continue

                    if name in seen_names:
                        warnings.append(
                            f"{functions_file}:{line_num}: duplicate tool name '{name}' "
                            f"(already defined in earlier subagent, skipping)"
                        )
                        continue

                    tools.append(tool)
                    seen_names.add(name)

        except OSError as e:
            warnings.append(f"cannot read {functions_file}: {e}")

    # Sort by tool name for stable output
    tools.sort(key=lambda t: t.get("name", ""))

    return tools, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate subagent functions.jsonl files into one combined file"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--subagents-dir", "-d", default=None,
        help="Subagents directory (default: ./subagents/ relative to this script)",
    )
    args = parser.parse_args()

    # Default subagents dir is sibling of this script
    if args.subagents_dir:
        subagents_dir = Path(args.subagents_dir)
    else:
        script_dir = Path(__file__).parent
        subagents_dir = script_dir / "subagents"
        # If running from subagents/ itself, look for sibling dirs
        if script_dir.name == "subagents":
            subagents_dir = script_dir

    tools, warnings = aggregate(subagents_dir)

    # Write output
    output_lines = [json.dumps(t, ensure_ascii=False) for t in tools]
    output_text = "\n".join(output_lines) + "\n" if output_lines else ""

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Aggregated {len(tools)} tools to {output_path}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)

    # Print warnings to stderr
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
