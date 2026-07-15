"""Wrapper 脚本端到端测试。

验证所有 wrapper 脚本的 stdin -> stdout JSON 链路正常工作。
不依赖外部服务（ChromaDB/LLM），只验证脚本能正确调用引擎并返回 JSON。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SUBAGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "subagents"
MTCLAW_ROOT = SUBAGENTS_DIR.parent


def run_wrapper(script_path: str, input_json: dict, env: dict = None) -> dict:
    """运行 wrapper 脚本，返回 JSON 输出。"""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(
        ["bash", script_path],
        input=json.dumps(input_json),
        capture_output=True,
        text=True,
        timeout=15,
        env=full_env,
    )
    if result.returncode != 0:
        pytest.fail(f"脚本执行失败: {script_path}\nstderr: {result.stderr}")

    return json.loads(result.stdout)


class TestRagWrappers:
    """测试 RAG wrapper 脚本。"""

    def test_rag_status(self, tmp_path):
        env = {"RAG_DATA_DIR": str(tmp_path / "rag_data")}
        script = str(SUBAGENTS_DIR / "rag" / "scripts" / "rag_status.sh")
        result = run_wrapper(script, {}, env)
        assert result["status"] == "ok"


class TestMemoryWrappers:
    """测试 Memory wrapper 脚本。"""

    def test_memory_remember(self, tmp_path):
        env = {"PROMETHEUS_DATA_DIR": str(tmp_path / "data")}
        script = str(SUBAGENTS_DIR / "memory" / "memory_remember.sh")
        result = run_wrapper(script, {
            "content": "hello from e2e",
            "type": "note",
        }, env)
        # memory_engine 返回 result=ok 或 error
        assert result.get("result") == "ok" or result.get("id") is not None or "error" not in result

    def test_memory_recall(self, tmp_path):
        env = {"PROMETHEUS_DATA_DIR": str(tmp_path / "data")}
        script = str(SUBAGENTS_DIR / "memory" / "memory_recall.sh")
        result = run_wrapper(script, {"query": "test"}, env)
        assert isinstance(result, dict)


class TestWritingWrappers:
    """测试 Writing wrapper 脚本。"""

    def test_writing_humanize_light(self, tmp_path):
        """humanize light 级别不需要 LLM。"""
        env = {
            "PROMETHEUS_DATA_DIR": str(tmp_path / "data"),
            "PROMETHEUS_TEMPLATES_DIR": str(tmp_path / "templates"),
        }
        script = str(SUBAGENTS_DIR / "writing" / "writing_humanize.sh")
        result = run_wrapper(script, {
            "text": "It is crucial to delve into the details.",
            "intensity": "light",
        }, env)
        # humanize light 返回可能不含 status 字段，检查没有 error 即可
        assert "error" not in result

    def test_writing_generate_missing_params(self, tmp_path):
        """缺少必填参数应返回错误 JSON，不应崩溃。"""
        env = {"PROMETHEUS_DATA_DIR": str(tmp_path / "data")}
        script = str(SUBAGENTS_DIR / "writing" / "writing_generate.sh")
        # wrapper 在缺少 doc_type 时用 error_exit 输出 JSON 但 exit 1
        import subprocess
        result = subprocess.run(
            ["bash", script],
            input=json.dumps({}),
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, **env},
        )
        # 应输出 JSON 错误（即使 exit code 非 0）
        try:
            output = json.loads(result.stdout)
            assert "error" in output
        except json.JSONDecodeError:
            pytest.fail(f"未输出有效 JSON: stdout={result.stdout}, stderr={result.stderr}")


class TestScheduleWrappers:
    """测试 Schedule wrapper 脚本。"""

    def test_schedule_create_event(self, tmp_path):
        env = {"PROMETHEUS_DATA_DIR": str(tmp_path / "data")}
        script = str(SUBAGENTS_DIR / "schedule" / "scripts" / "schedule_create_event.sh")
        result = run_wrapper(script, {
            "title": "测试会议",
            "start_time": "2026-07-20 14:00",
        }, env)
        assert result["status"] == "created"

    def test_schedule_query(self, tmp_path):
        env = {"PROMETHEUS_DATA_DIR": str(tmp_path / "data")}
        script = str(SUBAGENTS_DIR / "schedule" / "scripts" / "schedule_query.sh")
        result = run_wrapper(script, {"time_range": "all"}, env)
        assert result["status"] == "ok"

    def test_schedule_create_task(self, tmp_path):
        env = {"PROMETHEUS_DATA_DIR": str(tmp_path / "data")}
        script = str(SUBAGENTS_DIR / "schedule" / "scripts" / "schedule_create_task.sh")
        result = run_wrapper(script, {"title": "测试任务"}, env)
        assert result["status"] == "created"

    def test_schedule_list_tasks(self, tmp_path):
        env = {"PROMETHEUS_DATA_DIR": str(tmp_path / "data")}
        script = str(SUBAGENTS_DIR / "schedule" / "scripts" / "schedule_list_tasks.sh")
        result = run_wrapper(script, {}, env)
        assert result["status"] == "ok"

    def test_schedule_complete_task(self, tmp_path):
        env = {"PROMETHEUS_DATA_DIR": str(tmp_path / "data")}
        # 先创建任务
        create_script = str(SUBAGENTS_DIR / "schedule" / "scripts" / "schedule_create_task.sh")
        create_result = run_wrapper(create_script, {"title": "待完成"}, env)
        task_id = create_result["task_id"]
        # 再完成它
        script = str(SUBAGENTS_DIR / "schedule" / "scripts" / "schedule_complete_task.sh")
        result = run_wrapper(script, {"task_id": task_id}, env)
        assert result["status"] == "completed"


class TestChatWrapper:
    """测试 Chat wrapper 脚本。"""

    def test_chat_light_no_api(self, tmp_path):
        """没有配置路由模型时应返回 fallback 回复。"""
        env = {
            "ROUTING_URL": "",
            "ROUTING_MODEL": "",
            "PROMETHEUS_DATA_DIR": str(tmp_path / "data"),
        }
        script = str(SUBAGENTS_DIR / "chat" / "scripts" / "chat_light.sh")
        result = run_wrapper(script, {
            "mood": "casual",
            "_user_message": "你好",
        }, env)
        assert result["status"] == "ok"
        assert len(result["reply"]) > 0


class TestAggregateIntegration:
    """测试聚合脚本端到端。"""

    def test_aggregate_produces_valid_jsonl(self, tmp_path):
        """聚合脚本应输出有效 JSONL 文件。"""
        output_file = tmp_path / "functions.jsonl"
        result = subprocess.run(
            ["python3", str(SUBAGENTS_DIR / "aggregate_functions.py"),
             "--output", str(output_file)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"聚合失败: {result.stderr}"
        assert output_file.exists()

        # 验证每行是有效 JSON
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) >= 16  # 至少 16 个工具

        for line in lines:
            obj = json.loads(line.strip())
            assert "name" in obj
            assert "description" in obj
            assert "parameters" in obj
