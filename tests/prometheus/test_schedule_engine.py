"""schedule_engine 单元测试。

测试范围：
- parse_time 中文自然语言时间解析
- time_range_to_bounds 时间范围映射
- create_event / query_events CRUD
- create_task / list_tasks / complete_task CRUD
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SUBAGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "subagents"
sys.path.insert(0, str(SUBAGENTS_DIR / "schedule"))


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """使用临时数据库的 schedule_engine。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(data_dir))

    if "schedule_engine" in sys.modules:
        del sys.modules["schedule_engine"]
    import schedule_engine
    schedule_engine.DATA_DIR = str(data_dir)
    schedule_engine.DB_PATH = str(data_dir / "prometheus.db")
    return schedule_engine


class TestParseTime:
    """测试自然语言时间解析。"""

    def test_iso_format(self, engine):
        result = engine.parse_time("2026-07-15 10:00")
        assert result is not None
        assert "2026-07-15" in result

    def test_tomorrow(self, engine):
        result = engine.parse_time("明天")
        assert result is not None

    def test_tomorrow_with_time(self, engine):
        result = engine.parse_time("明天下午3点")
        assert result is not None

    def test_day_after_tomorrow(self, engine):
        result = engine.parse_time("后天")
        assert result is not None

    def test_next_week(self, engine):
        result = engine.parse_time("下周一")
        assert result is not None

    def test_next_week_with_time(self, engine):
        result = engine.parse_time("下周一上午10点")
        assert result is not None

    def test_n_days_later(self, engine):
        result = engine.parse_time("3天后")
        assert result is not None

    def test_n_hours_later(self, engine):
        result = engine.parse_time("2小时后")
        assert result is not None

    def test_invalid_input(self, engine):
        result = engine.parse_time("这不是一个时间")
        assert result is None

    def test_empty_input(self, engine):
        result = engine.parse_time("")
        assert result is None

    def test_none_input(self, engine):
        result = engine.parse_time(None)
        assert result is None


class TestExtractTime:
    """测试时间提取。"""

    def test_afternoon_time(self, engine):
        hour, minute = engine._extract_time("下午3点")
        assert hour == 15
        assert minute == 0

    def test_afternoon_time_with_minutes(self, engine):
        hour, minute = engine._extract_time("下午3点30分")
        assert hour == 15
        assert minute == 30

    def test_morning_time(self, engine):
        hour, minute = engine._extract_time("上午9点")
        assert hour == 9
        assert minute == 0

    def test_colon_format(self, engine):
        hour, minute = engine._extract_time("15:30")
        assert hour == 15
        assert minute == 30

    def test_no_time(self, engine):
        hour, minute = engine._extract_time("明天")
        assert hour is None
        assert minute is None


class TestTimeRangeToBounds:
    """测试时间范围映射。"""

    def test_today(self, engine):
        start, end = engine.time_range_to_bounds("today")
        assert start is not None
        assert end is not None
        assert start < end

    def test_tomorrow(self, engine):
        start, end = engine.time_range_to_bounds("tomorrow")
        assert start < end

    def test_this_week(self, engine):
        start, end = engine.time_range_to_bounds("this_week")
        assert start < end

    def test_next_week(self, engine):
        start, end = engine.time_range_to_bounds("next_week")
        assert start < end

    def test_all(self, engine):
        start, end = engine.time_range_to_bounds("all")
        assert start < end


class TestEventCRUD:
    """测试日程事件 CRUD。"""

    def test_create_event_basic(self, engine):
        result = engine.create_event({
            "title": "产品评审会",
            "start_time": "明天下午3点",
        })
        assert result["status"] == "created"
        assert result["event_id"] is not None
        assert result["title"] == "产品评审会"

    def test_create_event_with_location(self, engine):
        result = engine.create_event({
            "title": "团队会议",
            "start_time": "2026-07-20 14:00",
            "location": "会议室A",
            "category": "meeting",
        })
        assert result["status"] == "created"

    def test_create_event_missing_title(self, engine):
        result = engine.create_event({"start_time": "明天"})
        assert "error" in result

    def test_create_event_invalid_time(self, engine):
        result = engine.create_event({
            "title": "测试",
            "start_time": "不是时间",
        })
        assert "error" in result

    def test_query_events_today(self, engine):
        # 先创建一个今天的事件
        engine.create_event({
            "title": "今天的事",
            "start_time": datetime.now().isoformat(),
        })
        result = engine.query_events({"time_range": "today"})
        assert result["status"] == "ok"
        assert result["count"] >= 1

    def test_query_events_empty(self, engine):
        result = engine.query_events({"time_range": "today"})
        assert result["status"] == "ok"
        assert result["count"] == 0

    def test_query_events_all(self, engine):
        engine.create_event({"title": "A", "start_time": "2026-01-01 10:00"})
        engine.create_event({"title": "B", "start_time": "2026-12-31 10:00"})
        result = engine.query_events({"time_range": "all"})
        assert result["count"] >= 2


class TestTaskCRUD:
    """测试任务 CRUD。"""

    def test_create_task_basic(self, engine):
        result = engine.create_task({"title": "完成周报"})
        assert result["status"] == "created"
        assert result["task_id"] is not None

    def test_create_task_with_priority(self, engine):
        result = engine.create_task({
            "title": "紧急修复",
            "priority": 1,
            "due_date": "明天",
        })
        assert result["status"] == "created"

    def test_create_task_missing_title(self, engine):
        result = engine.create_task({"priority": 3})
        assert "error" in result

    def test_list_tasks_empty(self, engine):
        result = engine.list_tasks({})
        assert result["status"] == "ok"
        assert result["count"] == 0

    def test_list_tasks_after_create(self, engine):
        engine.create_task({"title": "任务A"})
        engine.create_task({"title": "任务B"})
        result = engine.list_tasks({})
        assert result["count"] >= 2

    def test_complete_task(self, engine):
        create_result = engine.create_task({"title": "待完成"})
        task_id = create_result["task_id"]
        result = engine.complete_task({"task_id": task_id})
        assert result["status"] == "completed"

    def test_complete_nonexistent_task(self, engine):
        result = engine.complete_task({"task_id": 99999})
        assert "error" in result

    def test_complete_task_missing_id(self, engine):
        result = engine.complete_task({})
        assert "error" in result

    def test_list_tasks_by_status(self, engine):
        engine.create_task({"title": "已完成任务"})
        # 完成它
        tasks = engine.list_tasks({"status": "pending"})
        if tasks["count"] > 0:
            task_id = tasks["tasks"][0]["id"]
            engine.complete_task({"task_id": task_id})
        # 查询已完成的
        result = engine.list_tasks({"status": "completed"})
        assert result["status"] == "ok"
