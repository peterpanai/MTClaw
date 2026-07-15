#!/usr/bin/env python3
"""Memory Engine — long-term memory & preferences for MTClaw subagents.

Architecture:
  - SQLite: 4 tables (memories, preferences, reminders, tags)
  - ChromaDB: memories collection for semantic vector search
  - dateparser: natural-language datetime parsing for reminders

Tables:
  memories     — episodic/factual memories with embeddings in ChromaDB
  preferences  — key-value user preferences (overwrite semantics)
  reminders    — scheduled reminders with natural-language due times
  tags         — tag index for memories (many-to-many)

CLI usage (called by wrapper scripts):
  python3 memory_engine.py remember    --content "..." [--type ...] [--tags ...] [--source ...]
  python3 memory_engine.py recall      --query "..." [--tags ...] [--type ...] [--limit N]
  python3 memory_engine.py set_reminder --title "..." --when "..." [--note "..."]
  python3 memory_engine.py list_reminders [--include_done]
  python3 memory_engine.py complete_reminder --id N
  python3 memory_engine.py set_pref    --key "..." --value "..."
  python3 memory_engine.py get_pref    --key "..."
  python3 memory_engine.py list_prefs
  python3 memory_engine.py get_memory  --id N
  python3 memory_engine.py delete_memory --id N

All output is JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dateparser

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(os.environ.get("MEMORY_DATA_DIR", str(SCRIPT_DIR / "data")))
DB_PATH = DEFAULT_DATA_DIR / "memory.db"
CHROMA_PATH = DEFAULT_DATA_DIR / "chroma"
CHROMA_COLLECTION = "memories"

VALID_MEMORY_TYPES = {"fact", "event", "preference", "note", "conversation"}


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    type        TEXT NOT NULL DEFAULT 'note',
    source      TEXT,
    tags        TEXT DEFAULT '[]',      -- JSON array of tag strings
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preferences (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    note        TEXT,
    due_at      TEXT NOT NULL,          -- ISO 8601 UTC
    raw_when    TEXT,                   -- original natural-language input
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | done | cancelled
    created_at  TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS tags (
    tag         TEXT NOT NULL,
    memory_id   TEXT NOT NULL,
    PRIMARY KEY (tag, memory_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_at);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
"""


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def init_db(data_dir: Path | None = None) -> Path:
    """Create data directory, SQLite database, and ChromaDB collection."""
    d = data_dir or DEFAULT_DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    global DB_PATH, CHROMA_PATH
    DB_PATH = d / "memory.db"
    CHROMA_PATH = d / "chroma"

    with _get_conn() as conn:
        conn.executescript(SCHEMA_SQL)

    _init_chroma()
    return d


@contextmanager
def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

_chroma_client = None
_chroma_collection = None


def _init_chroma():
    global _chroma_client, _chroma_collection
    if not HAS_CHROMADB:
        return
    _chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _chroma_add(memory_id: str, content: str, metadata: dict[str, Any]):
    if not _chroma_collection:
        return
    _chroma_collection.upsert(
        ids=[memory_id],
        documents=[content],
        metadatas=[metadata],
    )


def _chroma_query(query: str, n_results: int = 10, where: dict | None = None) -> list[dict[str, Any]]:
    if not _chroma_collection:
        return []
    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": min(n_results, _chroma_collection.count()) or n_results,
    }
    if where:
        kwargs["where"] = where
    try:
        results = _chroma_collection.query(**kwargs)
    except Exception:
        return []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return [
        {"id": ids[i], "content": docs[i], "metadata": metas[i], "distance": dists[i]}
        for i in range(len(ids))
    ]


def _chroma_delete(memory_id: str):
    if not _chroma_collection:
        return
    try:
        _chroma_collection.delete(ids=[memory_id])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Memory operations
# ---------------------------------------------------------------------------

def remember(
    content: str,
    memory_type: str = "note",
    tags: list[str] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Store a new memory in SQLite + ChromaDB."""
    if not content or not content.strip():
        return {"error": "content must not be empty"}
    memory_type = memory_type if memory_type in VALID_MEMORY_TYPES else "note"
    tags = tags or []
    now = _now_iso()
    memory_id = str(uuid.uuid4())

    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO memories (id, content, type, source, tags, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory_id, content.strip(), memory_type, source, json.dumps(tags), now, now),
        )
        for tag in tags:
            conn.execute(
                "INSERT OR IGNORE INTO tags (tag, memory_id) VALUES (?, ?)",
                (tag, memory_id),
            )

    _chroma_add(memory_id, content.strip(), {
        "type": memory_type,
        "source": source or "",
        "created_at": now,
    })

    return {
        "result": "ok",
        "id": memory_id,
        "content": content.strip(),
        "type": memory_type,
        "tags": tags,
        "source": source,
        "created_at": now,
    }


def recall(
    query: str,
    tags: list[str] | None = None,
    memory_type: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Semantic search across memories. Falls back to SQL LIKE if ChromaDB unavailable."""
    if not query or not query.strip():
        return {"error": "query must not be empty"}
    limit = max(1, min(limit, 50))

    # --- Semantic search via ChromaDB ---
    chroma_results = _chroma_query(query.strip(), n_results=limit * 2)
    matched_ids = [r["id"] for r in chroma_results]
    id_to_distance = {r["id"]: r["distance"] for r in chroma_results}

    # --- Fetch from SQLite ---
    with _get_conn() as conn:
        rows: list[sqlite3.Row] = []

        if matched_ids and HAS_CHROMADB:
            placeholders = ",".join("?" * len(matched_ids))
            sql = f"SELECT * FROM memories WHERE id IN ({placeholders})"
            params: list[Any] = list(matched_ids)
            if memory_type:
                sql += " AND type = ?"
                params.append(memory_type)
            rows = conn.execute(sql, params).fetchall()
            # Sort rows to match ChromaDB relevance order
            row_map = {row["id"]: row for row in rows}
            ordered: list[sqlite3.Row] = []
            for mid in matched_ids:
                if mid in row_map:
                    ordered.append(row_map[mid])
            rows = ordered
        else:
            # Fallback: SQL LIKE search
            sql = "SELECT * FROM memories WHERE content LIKE ?"
            params = [f"%{query.strip()}%"]
            if memory_type:
                sql += " AND type = ?"
                params.append(memory_type)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()

        # Tag filtering
        if tags:
            tag_set = set(tags)
            filtered = []
            for row in rows:
                row_tags = set(json.loads(row["tags"])) if row["tags"] else set()
                if tag_set & row_tags:
                    filtered.append(row)
            rows = filtered

        # Limit
        rows = rows[:limit]

        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "content": row["content"],
                "type": row["type"],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "source": row["source"],
                "created_at": row["created_at"],
                "relevance": round(1 - id_to_distance.get(row["id"], 1.0), 4),
            })

    return {
        "result": "ok",
        "query": query,
        "count": len(results),
        "memories": results,
    }


def get_memory(memory_id: str) -> dict[str, Any]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return {"error": "memory not found", "id": memory_id}
        return {
            "id": row["id"],
            "content": row["content"],
            "type": row["type"],
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "source": row["source"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def delete_memory(memory_id: str) -> dict[str, Any]:
    with _get_conn() as conn:
        row = conn.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return {"error": "memory not found", "id": memory_id}
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.execute("DELETE FROM tags WHERE memory_id = ?", (memory_id,))
    _chroma_delete(memory_id)
    return {"result": "ok", "deleted": memory_id}


# ---------------------------------------------------------------------------
# Preference operations
# ---------------------------------------------------------------------------

def set_pref(key: str, value: str) -> dict[str, Any]:
    if not key or not key.strip():
        return {"error": "key must not be empty"}
    now = _now_iso()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?",
            (key.strip(), value, now, value, now),
        )
    return {"result": "ok", "key": key.strip(), "value": value, "updated_at": now}


def get_pref(key: str) -> dict[str, Any]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM preferences WHERE key = ?", (key,)).fetchone()
        if not row:
            return {"error": "preference not found", "key": key}
        return {"key": row["key"], "value": row["value"], "updated_at": row["updated_at"]}


def list_prefs() -> dict[str, Any]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM preferences ORDER BY key").fetchall()
        prefs = [{"key": r["key"], "value": r["value"], "updated_at": r["updated_at"]} for r in rows]
    return {"result": "ok", "count": len(prefs), "preferences": prefs}


# ---------------------------------------------------------------------------
# Reminder operations
# ---------------------------------------------------------------------------

def parse_when(when_str: str) -> str | None:
    """Parse natural-language time string to ISO 8601 UTC."""
    if not when_str:
        return None
    # First try ISO format directly
    try:
        dt = datetime.fromisoformat(when_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    # Use dateparser for natural language
    parsed = dateparser.parse(
        when_str,
        settings={
            "TIMEZONE": "local",
            "TO_TIMEZONE": "UTC",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": datetime.now(timezone.utc),
        },
    )
    if parsed is None:
        return None
    return parsed.astimezone(timezone.utc).isoformat()


def set_reminder(title: str, when_str: str, note: str | None = None) -> dict[str, Any]:
    if not title or not title.strip():
        return {"error": "title must not be empty"}
    if not when_str or not when_str.strip():
        return {"error": "when must not be empty"}

    due_at = parse_when(when_str.strip())
    if not due_at:
        return {"error": f"could not parse time: {when_str}"}

    now = _now_iso()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO reminders (title, note, due_at, raw_when, status, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (title.strip(), note, due_at, when_str.strip(), now),
        )
        reminder_id = cur.lastrowid

    return {
        "result": "ok",
        "id": reminder_id,
        "title": title.strip(),
        "note": note,
        "due_at": due_at,
        "raw_when": when_str.strip(),
        "status": "pending",
        "created_at": now,
    }


def list_reminders(include_done: bool = False) -> dict[str, Any]:
    with _get_conn() as conn:
        if include_done:
            rows = conn.execute("SELECT * FROM reminders ORDER BY due_at ASC").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE status = 'pending' ORDER BY due_at ASC"
            ).fetchall()

        now = _now_iso()
        reminders = []
        for r in rows:
            reminders.append({
                "id": r["id"],
                "title": r["title"],
                "note": r["note"],
                "due_at": r["due_at"],
                "raw_when": r["raw_when"],
                "status": r["status"],
                "overdue": r["status"] == "pending" and r["due_at"] < now,
                "created_at": r["created_at"],
                "completed_at": r["completed_at"],
            })
    return {"result": "ok", "count": len(reminders), "reminders": reminders}


def complete_reminder(reminder_id: int) -> dict[str, Any]:
    now = _now_iso()
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
        if not row:
            return {"error": "reminder not found", "id": reminder_id}
        conn.execute(
            "UPDATE reminders SET status = 'done', completed_at = ? WHERE id = ?",
            (now, reminder_id),
        )
    return {"result": "ok", "id": reminder_id, "status": "done", "completed_at": now}


def get_due_reminders() -> dict[str, Any]:
    """Return all pending reminders whose due_at has passed."""
    now = _now_iso()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE status = 'pending' AND due_at <= ? ORDER BY due_at ASC",
            (now,),
        ).fetchall()
        reminders = []
        for r in rows:
            reminders.append({
                "id": r["id"],
                "title": r["title"],
                "note": r["note"],
                "due_at": r["due_at"],
                "raw_when": r["raw_when"],
                "status": r["status"],
                "created_at": r["created_at"],
            })
    return {"result": "ok", "count": len(reminders), "due_reminders": reminders, "checked_at": now}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_json(obj: dict[str, Any]):
    print(json.dumps(obj, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Memory engine for MTClaw — store, recall, and manage memories & reminders."
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Data directory for SQLite + ChromaDB (default: ./data)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # remember
    p_remember = sub.add_parser("remember", help="Store a new memory")
    p_remember.add_argument("--content", required=True, help="Memory content text")
    p_remember.add_argument("--type", default="note", choices=sorted(VALID_MEMORY_TYPES))
    p_remember.add_argument("--tags", nargs="*", default=[], help="Tags for this memory")
    p_remember.add_argument("--source", default=None, help="Source of the memory")

    # recall
    p_recall = sub.add_parser("recall", help="Search memories semantically")
    p_recall.add_argument("--query", required=True, help="Search query")
    p_recall.add_argument("--tags", nargs="*", default=[], help="Filter by tags")
    p_recall.add_argument("--type", default=None, choices=sorted(VALID_MEMORY_TYPES))
    p_recall.add_argument("--limit", type=int, default=10)

    # get_memory
    p_get = sub.add_parser("get_memory", help="Get a single memory by ID")
    p_get.add_argument("--id", required=True)

    # delete_memory
    p_del = sub.add_parser("delete_memory", help="Delete a memory by ID")
    p_del.add_argument("--id", required=True)

    # set_pref
    p_sp = sub.add_parser("set_pref", help="Set a preference")
    p_sp.add_argument("--key", required=True)
    p_sp.add_argument("--value", required=True)

    # get_pref
    p_gp = sub.add_parser("get_pref", help="Get a preference")
    p_gp.add_argument("--key", required=True)

    # list_prefs
    sub.add_parser("list_prefs", help="List all preferences")

    # set_reminder
    p_sr = sub.add_parser("set_reminder", help="Create a reminder")
    p_sr.add_argument("--title", required=True)
    p_sr.add_argument("--when", required=True, help="Natural-language time (e.g. 'tomorrow 9am', 'in 2 hours')")
    p_sr.add_argument("--note", default=None)

    # list_reminders
    p_lr = sub.add_parser("list_reminders", help="List reminders")
    p_lr.add_argument("--include_done", action="store_true")

    # complete_reminder
    p_cr = sub.add_parser("complete_reminder", help="Mark a reminder as done")
    p_cr.add_argument("--id", type=int, required=True)

    # get_due_reminders
    sub.add_parser("get_due_reminders", help="Get all due reminders")

    args = parser.parse_args()

    # Initialize DB
    data_dir = Path(args.data_dir) if args.data_dir else None
    init_db(data_dir)

    # Dispatch
    if args.command == "remember":
        _print_json(remember(args.content, args.type, args.tags, args.source))
    elif args.command == "recall":
        _print_json(recall(args.query, args.tags, args.type, args.limit))
    elif args.command == "get_memory":
        _print_json(get_memory(args.id))
    elif args.command == "delete_memory":
        _print_json(delete_memory(args.id))
    elif args.command == "set_pref":
        _print_json(set_pref(args.key, args.value))
    elif args.command == "get_pref":
        _print_json(get_pref(args.key))
    elif args.command == "list_prefs":
        _print_json(list_prefs())
    elif args.command == "set_reminder":
        _print_json(set_reminder(args.title, args.when, args.note))
    elif args.command == "list_reminders":
        _print_json(list_reminders(args.include_done))
    elif args.command == "complete_reminder":
        _print_json(complete_reminder(args.id))
    elif args.command == "get_due_reminders":
        _print_json(get_due_reminders())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
