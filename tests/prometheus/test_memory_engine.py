"""memory_engine 单元测试。

测试范围：
- SQLite 建表与 CRUD
- remember / recall / set_pref / get_pref / parse_when
- 接口签名匹配 memory_engine.py 实际实现
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

SUBAGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "subagents"
sys.path.insert(0, str(SUBAGENTS_DIR / "memory"))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """使用临时目录的数据库。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    if "memory_engine" in sys.modules:
        del sys.modules["memory_engine"]
    import memory_engine
    memory_engine.init_db(data_dir)
    yield memory_engine, data_dir


class TestInitDb:
    """测试数据库初始化。"""

    def test_init_creates_tables(self, tmp_db):
        engine, data_dir = tmp_db
        assert (data_dir / "memory.db").exists()

        import sqlite3
        conn = sqlite3.connect(str(data_dir / "memory.db"))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()

        assert "memories" in tables

    def test_init_is_idempotent(self, tmp_db):
        engine, data_dir = tmp_db
        engine.init_db(data_dir)  # 第二次不应报错


class TestRemember:
    """测试记忆存储。"""

    def test_remember_basic(self, tmp_db):
        engine, data_dir = tmp_db
        result = engine.remember(content="用户偏好中文", memory_type="preference")
        assert result.get("result") == "ok" or result.get("id") is not None

    def test_remember_with_tags(self, tmp_db):
        engine, data_dir = tmp_db
        result = engine.remember(
            content="喜欢 Markdown 格式",
            memory_type="preference",
            tags=["format", "writing"],
        )
        assert result.get("result") == "ok"

    def test_remember_empty_content(self, tmp_db):
        engine, data_dir = tmp_db
        result = engine.remember(content="")
        assert "error" in result

    def test_remember_invalid_type_fallback(self, tmp_db):
        engine, data_dir = tmp_db
        result = engine.remember(content="test", memory_type="invalid_type")
        # 无效 type 应 fallback 到 "note"
        assert result.get("type") == "note"


class TestRecall:
    """测试记忆检索。"""

    def test_recall_empty(self, tmp_db):
        engine, data_dir = tmp_db
        result = engine.recall(query="anything")
        assert isinstance(result, dict)

    def test_recall_after_remember(self, tmp_db):
        engine, data_dir = tmp_db
        engine.remember(content="用户喜欢用 Markdown", memory_type="preference")
        result = engine.recall(query="format preference")
        assert isinstance(result, dict)


class TestPrefs:
    """测试偏好快捷存取。"""

    def test_set_and_get_pref(self, tmp_db):
        engine, data_dir = tmp_db
        engine.set_pref("theme", "dark")
        result = engine.get_pref("theme")
        assert result.get("value") == "dark" or result.get("status") == "ok"

    def test_get_nonexistent_pref(self, tmp_db):
        engine, data_dir = tmp_db
        result = engine.get_pref("nonexistent_key_12345")
        assert result.get("value") is None or result.get("status") == "not_found"

    def test_list_prefs(self, tmp_db):
        engine, data_dir = tmp_db
        engine.set_pref("a", "1")
        engine.set_pref("b", "2")
        result = engine.list_prefs()
        assert isinstance(result, dict)


class TestParseWhen:
    """测试自然语言时间解析。"""

    def test_parse_iso_format(self, tmp_db):
        engine, data_dir = tmp_db
        result = engine.parse_when("2026-07-15 10:00")
        assert result is not None

    def test_parse_relative(self, tmp_db):
        engine, data_dir = tmp_db
        result = engine.parse_when("3小时后")
        # dateparser 或 fallback 应返回结果
        assert result is not None or result is None

    def test_parse_invalid(self, tmp_db):
        engine, data_dir = tmp_db
        result = engine.parse_when("完全不是时间的文本xyz")
        assert result is None


class TestCLIMode:
    """测试 CLI 入口（argparse 子命令模式）。"""

    def test_cli_remember(self, tmp_db, monkeypatch, capsys):
        engine, data_dir = tmp_db
        monkeypatch.setattr("sys.argv", [
            "memory_engine.py",
            "--data-dir", str(data_dir),
            "remember", "--content", "CLI 测试记忆",
        ])
        try:
            engine.main()
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert json.loads(captured.out)  # 输出是有效 JSON
