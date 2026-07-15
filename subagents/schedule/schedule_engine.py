#!/usr/bin/env python3
"""日程与任务 Subagent 引擎。

本地日程管理与任务追踪，支持自然语言时间解析。
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────

DATA_DIR = os.environ.get(
    "PROMETHEUS_DATA_DIR",
    os.path.expanduser("~/.function-router/prometheus/data"),
)
DB_PATH = os.path.join(DATA_DIR, "prometheus.db")

# ── dateparser 封装 ─────────────────────────────────

_dateparser = None

def parse_time(time_str: str) -> str | None:
    """解析中文自然语言时间，返回 ISO 8601 字符串。"""
    if not time_str or not time_str.strip():
        return None

    global _dateparser
    if _dateparser is None:
        try:
            import dateparser
            _dateparser = dateparser
        except ImportError:
            return _fallback_parse(time_str)

    # dateparser 用 languages 参数而非 settings 中的 LANGUAGE/LANGUAGES
    settings = {
        "PREFER_DATES_FROM": "future",
        "TIMEZONE": "Asia/Shanghai",
        "RETURN_AS_TIMEZONE_AWARE": True,
    }
    try:
        result = _dateparser.parse(time_str, languages=["zh"], settings=settings)
    except Exception:
        try:
            result = _dateparser.parse(time_str, settings=settings)
        except Exception:
            result = None

    if result:
        return result.isoformat()
    return _fallback_parse(time_str)


def _fallback_parse(time_str: str) -> str | None:
    """dateparser 不可用时的简单 fallback。"""
    now = datetime.now()

    # 中文数字映射
    cn_nums = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
               "两": 2, "明天": 1, "后天": 2, "大后天": 3,
               "今天": 0, "下周一": 7, "下周二": 8, "下周三": 9,
               "下周四": 10, "下周五": 11, "下周六": 12, "下周日": 13}

    s = time_str.strip()

    # 明天/后天
    if "明天" in s or "后天" in s:
        days = 1 if "明天" in s else 2
        target = now + timedelta(days=days)
        # 检查是否有时间
        hour, minute = _extract_time(s)
        if hour is not None:
            target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return target.isoformat()

    # 下周一/下周二...
    for day_name, days_ahead in [("下周一", 7), ("下周二", 8), ("下周三", 9),
                                  ("下周四", 10), ("下周五", 11), ("下周六", 12), ("下周日", 13)]:
        if day_name in s:
            today_weekday = now.weekday()  # 0=Monday
            target_weekday = days_ahead - 7  # 0=Monday
            days_until = (target_weekday - today_weekday + 7) % 7 + 7
            target = now + timedelta(days=days_until)
            hour, minute = _extract_time(s)
            if hour is not None:
                target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return target.isoformat()

    # N天后
    import re
    m = re.search(r"(\d+)天[后後]", s)
    if m:
        days = int(m.group(1))
        target = now + timedelta(days=days)
        return target.isoformat()

    # N小时后
    m = re.search(r"(\d+)小时[后後]", s)
    if m:
        hours = int(m.group(1))
        target = now + timedelta(hours=hours)
        return target.isoformat()

    # 下周
    if "下周" in s:
        target = now + timedelta(weeks=1)
        return target.date().isoformat()

    return None


def _extract_time(s: str) -> tuple[int | None, int | None]:
    """从字符串中提取时间（小时:分钟）。"""
    import re

    # "下午3点" / "下午3点30分"
    m = re.search(r"(上午|下午|早上|晚上)?(\d+)[点时](\d+)?分?", s)
    if m:
        period = m.group(1)
        hour = int(m.group(2))
        minute = int(m.group(3)) if m.group(3) else 0

        if period in ("下午", "晚上") and hour < 12:
            hour += 12
        elif period == "上午" and hour == 12:
            hour = 0

        return hour, minute

    # "15:30" 格式
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    return None, None


def time_range_to_bounds(time_range: str) -> tuple[str, str]:
    """将时间范围快捷词转换为 (start, end) ISO 8601 元组。"""
    now = datetime.now()

    if time_range == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif time_range == "tomorrow":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        end = start + timedelta(days=1)
    elif time_range == "this_week":
        today_weekday = now.weekday()  # 0=Monday
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=today_weekday)
        end = start + timedelta(days=7)
    elif time_range == "next_week":
        today_weekday = now.weekday()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=today_weekday) + timedelta(days=7)
        end = start + timedelta(days=7)
    else:  # all
        start = datetime(2000, 1, 1)
        end = datetime(2099, 12, 31)

    return start.isoformat(), end.isoformat()


# ── SQLite 操作 ──────────────────────────────────────

def get_db() -> sqlite3.Connection:
    """获取数据库连接，自动建表。"""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_tables(conn)
    return conn


def _init_tables(conn: sqlite3.Connection):
    """初始化表结构。"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            location TEXT,
            category TEXT DEFAULT 'general',
            reminder_minutes INTEGER DEFAULT 15,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now')),
            source TEXT DEFAULT 'user'
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            priority INTEGER DEFAULT 3,
            status TEXT DEFAULT 'pending',
            due_date TEXT,
            tags TEXT,
            parent_task_id INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            completed_at TEXT,
            FOREIGN KEY (parent_task_id) REFERENCES tasks(id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time);
        CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(due_date);
        CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
    """)
    conn.commit()


# ── 日程事件 CRUD ────────────────────────────────────

def create_event(params: dict) -> dict:
    """创建日程事件。"""
    title = params.get("title")
    if not title:
        return {"error": "title 是必填参数"}

    start_time_str = params.get("start_time", "")
    start_time = parse_time(start_time_str)
    if not start_time:
        return {"error": f"无法解析时间: {start_time_str}", "input": start_time_str}

    end_time = None
    if params.get("end_time"):
        end_time = parse_time(params["end_time"])

    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO events (title, description, start_time, end_time, location, category, reminder_minutes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (title, params.get("description"), start_time, end_time,
         params.get("location"), params.get("category", "general"),
         params.get("reminder_minutes", 15))
    )
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # 同步到记忆
    _sync_to_memory(f"日程: {title} at {start_time}")

    return {"status": "created", "event_id": event_id, "title": title,
            "start_time": start_time, "end_time": end_time}


def query_events(params: dict) -> dict:
    """查询日程事件。"""
    time_range = params.get("time_range", "today")
    category = params.get("category")
    status = params.get("status", "pending")

    start, end = time_range_to_bounds(time_range)

    conn = get_db()
    sql = "SELECT * FROM events WHERE start_time >= ? AND start_time < ?"
    args = [start, end]

    if status and status != "all":
        sql += " AND status = ?"
        args.append(status)
    if category:
        sql += " AND category = ?"
        args.append(category)

    sql += " ORDER BY start_time ASC"
    rows = conn.execute(sql, args).fetchall()
    conn.close()

    events = [dict(r) for r in rows]
    return {"status": "ok", "count": len(events), "events": events,
            "time_range": time_range}


# ── 任务 CRUD ────────────────────────────────────────

def create_task(params: dict) -> dict:
    """创建待办任务。"""
    title = params.get("title")
    if not title:
        return {"error": "title 是必填参数"}

    due_date = None
    if params.get("due_date"):
        due_date = parse_time(params["due_date"])

    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO tasks (title, description, priority, due_date, tags)
           VALUES (?, ?, ?, ?, ?)""",
        (title, params.get("description"), params.get("priority", 3),
         due_date, params.get("tags"))
    )
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()

    _sync_to_memory(f"任务: {title}, 优先级: {params.get('priority', 3)}")

    return {"status": "created", "task_id": task_id, "title": title,
            "due_date": due_date}


def list_tasks(params: dict) -> dict:
    """列出待办任务。"""
    status = params.get("status", "pending")
    priority = params.get("priority")
    tags = params.get("tags")

    conn = get_db()
    sql = "SELECT * FROM tasks WHERE 1=1"
    args = []

    if status and status != "all":
        sql += " AND status = ?"
        args.append(status)
    if priority:
        sql += " AND priority = ?"
        args.append(priority)
    if tags:
        sql += " AND tags LIKE ?"
        args.append(f"%{tags}%")

    sql += " ORDER BY priority ASC, due_date ASC"
    rows = conn.execute(sql, args).fetchall()
    conn.close()

    tasks = [dict(r) for r in rows]
    return {"status": "ok", "count": len(tasks), "tasks": tasks}


def complete_task(params: dict) -> dict:
    """标记任务完成。"""
    task_id = params.get("task_id")
    if not task_id:
        return {"error": "task_id 是必填参数"}

    conn = get_db()
    cursor = conn.execute(
        "UPDATE tasks SET status = 'completed', completed_at = datetime('now') WHERE id = ?",
        (task_id,)
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()

    if affected == 0:
        return {"error": f"未找到 task_id={task_id} 的任务"}

    return {"status": "completed", "task_id": task_id}


# ── 记忆协同 ─────────────────────────────────────────

def _sync_to_memory(content: str):
    """同步到记忆 Subagent（best-effort，失败不影响主流程）。"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory"))
        from memory_engine import remember
        remember({"category": "note", "key": f"schedule_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                  "value": content, "importance": 2})
    except Exception:
        pass


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

    commands = {
        "create_event": create_event,
        "query": query_events,
        "create_task": create_task,
        "list_tasks": list_tasks,
        "complete_task": complete_task,
    }

    handler = commands.get(command)
    if handler:
        result = handler(params)
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps({"error": f"未知命令: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
