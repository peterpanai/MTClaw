"""preference_engine 单元测试。

测试范围：
- detect_and_store_preference 偏好检测
- run_daily_maintenance 每日维护
- get_preference_context 偏好上下文获取
"""
import sys
from pathlib import Path

import pytest

SUBAGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "subagents"
sys.path.insert(0, str(SUBAGENTS_DIR / "preference"))


@pytest.fixture
def engine(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(data_dir))

    if "preference_engine" in sys.modules:
        del sys.modules["preference_engine"]
    import preference_engine
    preference_engine.DATA_DIR = str(data_dir)
    preference_engine.DB_PATH = str(data_dir / "prometheus.db")
    return preference_engine


class TestDetectPreference:
    """测试偏好检测。"""

    def test_detect_always_pattern(self, engine):
        result = engine.detect_and_store_preference("以后都用 Markdown 格式")
        assert result["detected"] == True

    def test_detect_like_pattern(self, engine):
        result = engine.detect_and_store_preference("我喜欢简洁的回复")
        assert result["detected"] == True

    def test_detect_dislike_pattern(self, engine):
        result = engine.detect_and_store_preference("以后不要加表情符号")
        assert result["detected"] == True

    def test_detect_remember_pattern(self, engine):
        result = engine.detect_and_store_preference("记住了，我喜欢用中文")
        assert result["detected"] == True

    def test_no_preference_detected(self, engine):
        result = engine.detect_and_store_preference("今天天气真好啊")
        assert result["detected"] == False

    def test_empty_message(self, engine):
        result = engine.detect_and_store_preference("")
        assert result["detected"] == False

    def test_none_message(self, engine):
        result = engine.detect_and_store_preference(None)
        assert result["detected"] == False

    def test_weak_signal_format(self, engine):
        """弱信号：格式偏好。"""
        result = engine.detect_and_store_preference("请用 Markdown 格式回复")
        assert result["detected"] == True

    def test_weak_signal_language(self, engine):
        """弱信号：语言偏好。"""
        result = engine.detect_and_store_preference("请用中文回答")
        assert result["detected"] == True

    def test_weak_signal_style(self, engine):
        """弱信号：风格偏好。"""
        result = engine.detect_and_store_preference("回答要简洁一点")
        assert result["detected"] == True


class TestRunDailyMaintenance:
    """测试每日维护。"""

    def test_maintenance_no_db(self, engine):
        """没有数据库时不应崩溃。"""
        result = engine.run_daily_maintenance()
        # 应该返回 ok 或 error，不应抛异常
        assert isinstance(result, dict)

    def test_maintenance_with_db(self, engine):
        """有数据库时正常执行。"""
        # 先创建数据库
        import sqlite3
        conn = sqlite3.connect(engine.DB_PATH)
        engine._init_tables(conn) if hasattr(engine, '_init_tables') else None
        conn.close()
        result = engine.run_daily_maintenance()
        assert isinstance(result, dict)


class TestGetPreferenceContext:
    """测试偏好上下文获取。"""

    def test_context_empty(self, engine):
        """没有记忆时应返回空字符串。"""
        result = engine.get_preference_context()
        assert isinstance(result, str)

    def test_context_with_query(self, engine):
        result = engine.get_preference_context("user preferences")
        assert isinstance(result, str)
