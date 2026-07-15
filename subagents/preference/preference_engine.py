#!/usr/bin/env python3
"""即时偏好引擎。

检测用户偏好声明，实时写入记忆。实现"越用越懂你"的核心差异化。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.expanduser("~/.prometheus/data"),
)
DB_PATH = os.path.join(DATA_DIR, "prometheus.db")

# ── 偏好检测模式 ─────────────────────────────────────

# 强偏好声明（高置信度）
STRONG_PATTERNS = [
    # "以后都XXX" / "以后总是XXX"
    (re.compile(r"以后(?:都|总是|一律)(.+)"), "always"),
    # "记住了，我喜欢XXX"
    (re.compile(r"记住了[，,。]?我(?:喜欢|偏好|习惯)(.+)"), "like"),
    # "我喜欢XXX" / "我偏好XXX"
    (re.compile(r"我(?:喜欢|偏好|习惯)(.+)"), "like"),
    # "不要XXX" / "以后不要XXX"
    (re.compile(r"(?:以后)?不(?:要|要|喜欢)(.+)"), "dislike"),
    # "总是XXX" / "永远XXX"
    (re.compile(r"(?:总是|永远|每次都)(.+)"), "always"),
    # "我的XXX是XXX"
    (re.compile(r"我(?:的)?(.+?)(?:是|为)(.+)"), "fact"),
]

# 弱偏好信号（需要累积确认）
WEAK_SIGNALS = {
    "format": [re.compile(r"(?:用|使用|格式)(?:Markdown|markdown|MD|md|纯文本|JSON|json)")],
    "language": [re.compile(r"(?:用|使用)(?:中文|英文|日语|韩语)")],
    "style": [re.compile(r"(?:简洁|详细|正式|随意|口语化|书面)")],
}


def detect_and_store_preference(user_message: str) -> dict:
    """检测用户消息中的偏好声明，实时写入记忆。

    Args:
        user_message: 用户原始消息

    Returns:
        检测结果，包含是否检测到偏好、偏好内容等
    """
    if not user_message or not user_message.strip():
        return {"detected": False}

    detected = []

    # 1. 强偏好模式匹配
    for pattern, pref_type in STRONG_PATTERNS:
        match = pattern.search(user_message)
        if match:
            content = match.group(1).strip() if match.lastindex >= 1 else ""
            if len(content) < 100 and content:  # 避免匹配到过长的文本
                detected.append({
                    "type": pref_type,
                    "content": content,
                    "confidence": "high",
                    "source": "explicit_declaration",
                })

    # 2. 弱信号检测
    for signal_type, patterns in WEAK_SIGNALS.items():
        for pattern in patterns:
            match = pattern.search(user_message)
            if match:
                detected.append({
                    "type": signal_type,
                    "content": match.group(0),
                    "confidence": "low",
                    "source": "implicit_signal",
                })

    if not detected:
        return {"detected": False}

    # 3. 写入记忆
    stored = []
    for pref in detected:
        try:
            result = _store_preference(pref)
            if result:
                stored.append(result)
        except Exception as e:
            # 记忆写入失败不影响主流程
            pass

    return {
        "detected": True,
        "count": len(stored),
        "preferences": stored,
    }


def _store_preference(pref: dict) -> dict | None:
    """将偏好写入 memory_engine 的 SQLite + ChromaDB。"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
        from memory_engine import remember
        category = f"preference_{pref['type']}"
        key = f"pref_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        value = pref["content"]
        importance = 5 if pref["confidence"] == "high" else 2

        result = remember({
            "category": category,
            "key": key,
            "value": value,
            "importance": importance,
        })

        return {
            "category": category,
            "value": value,
            "importance": importance,
            "stored": True,
        }
    except Exception:
        return None


def run_daily_maintenance() -> dict:
    """每日记忆维护：衰减/强化 + 交互统计。

    应通过 cron 每日凌晨 2:00 执行。
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # 1. 衰减：30天未访问的记忆 importance -1（最低 1）
        conn.execute("""
            UPDATE memories
            SET importance = MAX(1, importance - 1)
            WHERE access_count = 0
              AND datetime(updated_at) < datetime('now', '-30 days')
              AND importance > 1
        """)

        # 2. 强化：高频访问的记忆 importance +1（最高 5）
        conn.execute("""
            UPDATE memories
            SET importance = MIN(5, importance + 1)
            WHERE access_count >= 5
              AND importance < 5
        """)

        # 3. 重置 access_count
        conn.execute("UPDATE memories SET access_count = 0")

        # 4. 统计
        stats = {
            "total_memories": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            "total_events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] if _table_exists(conn, "events") else 0,
            "total_tasks": conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] if _table_exists(conn, "tasks") else 0,
            "interaction_count": conn.execute("SELECT COUNT(*) FROM interaction_log").fetchone()[0] if _table_exists(conn, "interaction_log") else 0,
        }

        conn.commit()
        conn.close()

        return {"status": "ok", "maintenance": "completed", "stats": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """检查表是否存在。"""
    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    ).fetchone()
    return result is not None


def get_preference_context(context: str = "", top_k: int = 5) -> str:
    """获取偏好上下文，用于注入到 system prompt。"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
        from memory_engine import recall
        result = recall({"context": context or "user preferences", "top_k": top_k})
        if result and isinstance(result, dict) and result.get("memories"):
            lines = []
            for mem in result["memories"]:
                lines.append(f"- {mem.get('key', '')}: {mem.get('value', '')}")
            return "\n".join(lines)
    except Exception:
        pass
    return ""


# ── CLI 入口 ──────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "缺少命令参数"}))
        sys.exit(1)

    command = sys.argv[1]

    try:
        params = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        params = {}

    if command == "detect":
        message = params.get("message", "")
        result = detect_and_store_preference(message)
        print(json.dumps(result, ensure_ascii=False))

    elif command == "maintenance":
        result = run_daily_maintenance()
        print(json.dumps(result, ensure_ascii=False))

    elif command == "context":
        context = params.get("context", "")
        top_k = params.get("top_k", 5)
        result = get_preference_context(context, top_k)
        print(json.dumps({"context": result}, ensure_ascii=False))

    else:
        print(json.dumps({"error": f"未知命令: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
