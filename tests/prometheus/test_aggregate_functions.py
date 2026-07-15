"""aggregate_functions 单元测试。

测试范围：
- 聚合所有 subagents/*/functions.jsonl
- 工具名去重
- 输出格式正确
"""
import json
import sys
from pathlib import Path

import pytest

SUBAGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "subagents"
sys.path.insert(0, str(SUBAGENTS_DIR))


@pytest.fixture
def agg():
    if "aggregate_functions" in sys.modules:
        del sys.modules["aggregate_functions"]
    import aggregate_functions
    return aggregate_functions


class TestAggregate:
    """测试工具聚合。"""

    def test_aggregate_returns_list(self, agg):
        tools, warnings = agg.aggregate(SUBAGENTS_DIR)
        assert isinstance(tools, list)
        assert isinstance(warnings, list)

    def test_aggregate_has_all_subagents(self, agg):
        tools, _ = agg.aggregate(SUBAGENTS_DIR)
        tool_names = {t["name"] for t in tools}
        # 应包含所有 5 个 Subagent 的工具
        assert "rag_search" in tool_names
        assert "memory_remember" in tool_names
        assert "writing_generate" in tool_names
        assert "schedule_create_event" in tool_names
        assert "chat_light" in tool_names

    def test_aggregate_tool_count(self, agg):
        tools, _ = agg.aggregate(SUBAGENTS_DIR)
        # 至少 16 个工具
        assert len(tools) >= 16

    def test_aggregate_each_tool_has_required_fields(self, agg):
        tools, _ = agg.aggregate(SUBAGENTS_DIR)
        for tool in tools:
            assert "name" in tool, f"工具缺少 name 字段"
            assert "description" in tool, f"工具 {tool.get('name')} 缺少 description"
            assert "parameters" in tool, f"工具 {tool.get('name')} 缺少 parameters"

    def test_aggregate_no_duplicates(self, agg):
        tools, warnings = agg.aggregate(SUBAGENTS_DIR)
        names = [t["name"] for t in tools]
        duplicates = [n for n in names if names.count(n) > 1]
        assert len(duplicates) == 0, f"发现重复工具: {duplicates}"

    def test_aggregate_writing_tools(self, agg):
        tools, _ = agg.aggregate(SUBAGENTS_DIR)
        names = {t["name"] for t in tools}
        assert "writing_generate" in names
        assert "writing_polish" in names
        assert "writing_translate" in names
        assert "writing_humanize" in names

    def test_aggregate_schedule_tools(self, agg):
        tools, _ = agg.aggregate(SUBAGENTS_DIR)
        names = {t["name"] for t in tools}
        assert "schedule_create_event" in names
        assert "schedule_query" in names
        assert "schedule_create_task" in names
        assert "schedule_list_tasks" in names
        assert "schedule_complete_task" in names

    def test_aggregate_rag_tools(self, agg):
        tools, _ = agg.aggregate(SUBAGENTS_DIR)
        names = {t["name"] for t in tools}
        assert "rag_search" in names
        assert "rag_ingest" in names
        assert "rag_status" in names

    def test_aggregate_memory_tools(self, agg):
        tools, _ = agg.aggregate(SUBAGENTS_DIR)
        names = {t["name"] for t in tools}
        assert "memory_remember" in names
        assert "memory_recall" in names
        assert "memory_set_reminder" in names

    def test_aggregate_output_is_valid_jsonl(self, agg):
        """聚合输出应为有效 JSONL（每行一个 JSON 对象）。"""
        import io
        import tempfile

        tools, _ = agg.aggregate(SUBAGENTS_DIR)
        # 模拟输出
        lines = [json.dumps(t, ensure_ascii=False) for t in tools]
        for line in lines:
            obj = json.loads(line)  # 每行应是有效 JSON
            assert "name" in obj
