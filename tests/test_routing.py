import json
from pathlib import Path

import httpx
import pytest

from function_router import server


@pytest.fixture(autouse=True)
def restore_state() -> None:
    old_config = server.STATE.config
    old_tools = server.STATE.tools
    old_http_client = server.STATE.http_client
    old_config_path = server.STATE.config_path
    old_saved_contexts = dict(server._QWEN_SAVED_CONTEXTS)
    old_pending_turns = dict(server._QWEN_PENDING_UPSTREAM_TURNS)
    old_pending_delegated = {
        key: set(value)
        for key, value in server._SESSION_PENDING_DELEGATED_TOOL_IDS.items()
    }
    old_tool_history = list(server.TOOL_HISTORY)
    try:
        server._QWEN_SAVED_CONTEXTS.clear()
        server._QWEN_PENDING_UPSTREAM_TURNS.clear()
        server._SESSION_PENDING_DELEGATED_TOOL_IDS.clear()
        server.TOOL_HISTORY.clear()
        yield
    finally:
        server.STATE.config = old_config
        server.STATE.tools = old_tools
        server.STATE.http_client = old_http_client
        server.STATE.config_path = old_config_path
        server._QWEN_SAVED_CONTEXTS.clear()
        server._QWEN_SAVED_CONTEXTS.update(old_saved_contexts)
        server._QWEN_PENDING_UPSTREAM_TURNS.clear()
        server._QWEN_PENDING_UPSTREAM_TURNS.update(old_pending_turns)
        server._SESSION_PENDING_DELEGATED_TOOL_IDS.clear()
        server._SESSION_PENDING_DELEGATED_TOOL_IDS.update(old_pending_delegated)
        server.TOOL_HISTORY.clear()
        server.TOOL_HISTORY.extend(old_tool_history)


def make_config(max_tool_rounds: int = 3) -> server.AppConfig:
    return server.AppConfig(
        listen_host="127.0.0.1",
        listen_port=18790,
        routing=server.ModelConfig("http://router/v1", "router", "any"),
        upstream=server.ModelConfig("http://upstream/v1", "upstream", "secret"),
        functions_file="functions.jsonl",
        scripts_dir="scripts",
        max_tool_rounds=max_tool_rounds,
        tool_exec_timeout_s=30,
        root_dir=server.Path("/tmp/function-router-tests"),
        config_path=server.Path("/tmp/function-router-tests/config.json"),
    )


def make_temp_config(root_dir: Path, max_tool_rounds: int = 3) -> server.AppConfig:
    return server.AppConfig(
        listen_host="127.0.0.1",
        listen_port=18790,
        routing=server.ModelConfig("http://router/v1", "router", "any"),
        upstream=server.ModelConfig("http://upstream/v1", "upstream", "secret"),
        functions_file="functions.jsonl",
        scripts_dir="scripts",
        max_tool_rounds=max_tool_rounds,
        tool_exec_timeout_s=30,
        root_dir=root_dir,
        config_path=root_dir / "config.json",
    )


def tool_definition(name: str = "custom_tool") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def qwen_tool_call(
    name: str = "custom_tool",
    *,
    call_id: str = "call_1",
    arguments=None,
) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": {"x": 1} if arguments is None else arguments,
        },
    }


def write_stub_script(root_dir: Path, name: str = "custom_tool", marker: Path | None = None) -> None:
    scripts_dir = root_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    marker_line = f"printf ran > {marker}\n" if marker is not None else ""
    script = scripts_dir / f"{name}.sh"
    script.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "cat >/dev/null\n"
        f"{marker_line}"
        "printf '{\"ok\":true}'\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def sse_payloads(text: str) -> list[dict]:
    payloads = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            continue
        payloads.append(json.loads(data))
    return payloads


async def asgi_request(method: str, url: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, url, **kwargs)


@pytest.mark.asyncio
async def test_run_tool_loop_returns_no_tool_context_when_no_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server.STATE.config = make_config()
    server.STATE.tools = [{"type": "function", "function": {"name": "system_control"}}]

    async def fake_call_qwen(messages):
        return {"choices": [{"message": {"content": "cannot handle"}}]}

    monkeypatch.setattr(server, "call_qwen", fake_call_qwen)

    result = await server.run_tool_loop("hello")

    assert result.used_any_tool is False
    assert result.last_function_name is None
    assert result.tool_context == []
    assert result.max_rounds_exhausted is False


@pytest.mark.asyncio
async def test_run_tool_loop_single_tool_call_normalizes_null_content_and_missing_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server.STATE.config = make_config()
    server.STATE.tools = [{"type": "function", "function": {"name": "system_control"}}]
    calls = [
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "system_control",
                                    "arguments": '{"category":"volume","action":"status"}',
                                }
                            }
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "done"}}]},
    ]

    async def fake_call_qwen(messages):
        return calls.pop(0)

    async def fake_execute_tool(function_name: str, arguments_json: str):
        assert function_name == "system_control"
        assert '"status"' in arguments_json
        return {"success": True, "current_value": 50}

    monkeypatch.setattr(server, "call_qwen", fake_call_qwen)
    monkeypatch.setattr(server, "execute_tool", fake_execute_tool)

    result = await server.run_tool_loop("check volume")

    assert result.used_any_tool is True
    assert result.last_function_name == "system_control"
    assert result.tool_rounds == 2
    assert result.max_rounds_exhausted is False
    assert result.tool_context[0] == {"role": "user", "content": "check volume"}
    assistant_context = result.tool_context[1]
    assert assistant_context["role"] == "assistant"
    assert assistant_context["content"] == ""
    assert assistant_context["tool_calls"][0]["function"] == {
        "name": "system_control",
        "arguments": '{"category":"volume","action":"status"}',
    }
    tool_context = result.tool_context[2]
    assert tool_context["role"] == "tool"
    assert tool_context["tool_call_id"] == "call_system_control_1"
    assert tool_context["name"] == "system_control"
    assert tool_context["content"] == '{"success": true, "current_value": 50}'


@pytest.mark.asyncio
async def test_run_tool_loop_multi_round_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    server.STATE.config = make_config(max_tool_rounds=3)
    server.STATE.tools = [{"type": "function", "function": {"name": "system_control"}}]
    calls = [
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_one",
                                "function": {
                                    "name": "system_control",
                                    "arguments": '{"category":"brightness","action":"status"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_two",
                                "function": {
                                    "name": "system_control",
                                    "arguments": '{"category":"brightness","action":"set","value":80}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "done"}}]},
    ]
    executed = []

    async def fake_call_qwen(messages):
        return calls.pop(0)

    async def fake_execute_tool(function_name: str, arguments_json: str):
        executed.append((function_name, arguments_json))
        return {"ok": True}

    monkeypatch.setattr(server, "call_qwen", fake_call_qwen)
    monkeypatch.setattr(server, "execute_tool", fake_execute_tool)

    result = await server.run_tool_loop("make it brighter")

    assert result.used_any_tool is True
    assert result.tool_rounds == 3
    assert result.max_rounds_exhausted is False
    assert executed == [
        ("system_control", '{"category":"brightness","action":"status"}'),
        ("system_control", '{"category":"brightness","action":"set","value":80}'),
    ]
    assert [message["role"] for message in result.tool_context] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


@pytest.mark.asyncio
async def test_run_tool_loop_returns_context_when_max_rounds_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server.STATE.config = make_config(max_tool_rounds=2)
    server.STATE.tools = [{"type": "function", "function": {"name": "wallpaper_control"}}]

    async def fake_call_qwen(messages):
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_loop",
                                "function": {
                                    "name": "wallpaper_control",
                                    "arguments": '{"action":"random"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

    async def fake_execute_tool(function_name: str, arguments_json: str):
        return {"success": True}

    monkeypatch.setattr(server, "call_qwen", fake_call_qwen)
    monkeypatch.setattr(server, "execute_tool", fake_execute_tool)

    result = await server.run_tool_loop("random wallpaper")

    assert result.used_any_tool is True
    assert result.last_function_name == "wallpaper_control"
    assert result.tool_rounds == 2
    assert result.max_rounds_exhausted is True
    assert result.tool_context[-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_execute_tool_builtin_find_does_not_require_script_file(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "movie.mp4").write_text("", encoding="utf-8")

    server.STATE.config = make_temp_config(tmp_path)

    result = await server.execute_tool(
        "find",
        json.dumps({"path": str(media_dir), "name_pattern": "*.mp4"}),
    )

    assert result["result"] == "ok"
    assert str(media_dir / "movie.mp4") in result["tool_output"]


@pytest.mark.asyncio
async def test_execute_tool_builtin_find_expands_tilde_to_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    home_dir = tmp_path / "home"
    (home_dir / "music").mkdir(parents=True)
    (home_dir / "music" / "song.mp3").write_text("", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home_dir))

    server.STATE.config = make_temp_config(tmp_path)

    result = await server.execute_tool(
        "find",
        json.dumps({"path": "~", "name_pattern": "*.mp3", "entry_type": "file"}),
    )

    assert result["result"] == "ok"
    assert str(home_dir / "music" / "song.mp3") in result["matches"]


@pytest.mark.asyncio
async def test_execute_tool_builtin_find_reports_missing_path_error(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    server.STATE.config = make_temp_config(tmp_path)

    result = await server.execute_tool(
        "find",
        json.dumps({"path": str(tmp_path / "missing"), "name_pattern": "*.mp3"}),
    )

    assert result["error"]
    assert result["returncode"] != 0


@pytest.mark.asyncio
async def test_execute_tool_builtin_cat_returns_file_content(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    note_path = tmp_path / "note.txt"
    note_path.write_text("alpha\nbeta\n", encoding="utf-8")

    server.STATE.config = make_temp_config(tmp_path)

    result = await server.execute_tool(
        "cat",
        json.dumps({"path": str(note_path)}),
    )

    assert result["result"] == "ok"
    assert result["tool_output"] == "alpha\nbeta"


@pytest.mark.asyncio
async def test_execute_tool_builtin_ls_lists_directory_contents(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "a.mp4").write_text("", encoding="utf-8")
    (media_dir / "b.txt").write_text("", encoding="utf-8")

    server.STATE.config = make_temp_config(tmp_path)

    result = await server.execute_tool("ls", json.dumps({"path": str(media_dir)}))

    assert result["result"] == "ok"
    assert result["entries"] == ["a.mp4", "b.txt"]


@pytest.mark.asyncio
async def test_execute_tool_builtin_grep_matches_text(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    note_path = tmp_path / "note.txt"
    note_path.write_text("alpha\nbeta keyword\n", encoding="utf-8")

    server.STATE.config = make_temp_config(tmp_path)

    result = await server.execute_tool(
        "grep",
        json.dumps({"path": str(note_path), "pattern": "keyword"}),
    )

    assert result["result"] == "ok"
    assert result["matches"] == [f"2:beta keyword"]


@pytest.mark.asyncio
async def test_execute_tool_builtin_sleep_validates_seconds(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    server.STATE.config = make_temp_config(tmp_path)

    result = await server.execute_tool("sleep", json.dumps({"seconds": -1}))

    assert result["error"] == "seconds must be a non-negative number"


def test_system_prompt_requires_verbatim_copy_of_tool_values() -> None:
    prompt = server.SYSTEM_PROMPT
    assert "copy that value verbatim exactly as it appears" in prompt
    assert "Never replace any part of an exact value with ellipsis" in prompt
    assert "If the exact value is missing or ambiguous, do not guess" in prompt
    assert "Never substitute visually similar Unicode characters" in prompt
    assert "Cyrillic or Greek letters" in prompt


@pytest.mark.asyncio
async def test_execute_tool_non_builtin_still_requires_script_file(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    server.STATE.config = make_temp_config(tmp_path)

    result = await server.execute_tool("custom_tool", json.dumps({"path": "/tmp"}))

    assert result["error"] == "script not found: custom_tool.sh"


class _StubResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _StubHttpClient:
    """Records call timeouts and replays a scripted sequence of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.timeouts_seen: list[float] = []

    async def post(self, url, json=None, headers=None, timeout=None):
        import httpx

        self.call_count += 1
        self.timeouts_seen.append(timeout)
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _StubResponse(item)


@pytest.mark.asyncio
async def test_call_qwen_uses_configured_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config()
    config.routing_timeout_s = 15.0
    server.STATE.config = config
    server.STATE.tools = []
    client = _StubHttpClient([{"choices": [{"message": {"content": "ok"}}]}])
    server.STATE.http_client = client

    await server.call_qwen([{"role": "user", "content": "hi"}])

    assert client.call_count == 1
    assert client.timeouts_seen == [15.0]


@pytest.mark.asyncio
async def test_call_qwen_retries_once_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    server.STATE.config = make_config()
    server.STATE.tools = []
    client = _StubHttpClient([
        httpx.ReadTimeout("slow"),
        {"choices": [{"message": {"content": "recovered"}}]},
    ])
    server.STATE.http_client = client
    monkeypatch.setattr(server.asyncio, "sleep", lambda _s: _noop_sleep())

    result = await server.call_qwen([{"role": "user", "content": "hi"}])

    assert client.call_count == 2
    assert result["choices"][0]["message"]["content"] == "recovered"


@pytest.mark.asyncio
async def test_call_qwen_propagates_timeout_after_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    server.STATE.config = make_config()
    server.STATE.tools = []
    client = _StubHttpClient([
        httpx.ReadTimeout("slow"),
        httpx.ReadTimeout("still slow"),
    ])
    server.STATE.http_client = client
    monkeypatch.setattr(server.asyncio, "sleep", lambda _s: _noop_sleep())

    with pytest.raises(httpx.TimeoutException):
        await server.call_qwen([{"role": "user", "content": "hi"}])

    assert client.call_count == 2


async def _noop_sleep() -> None:
    return None


@pytest.mark.asyncio
async def test_chat_delegated_first_turn_returns_tool_calls_without_running_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "script-ran"
    write_stub_script(tmp_path, marker=marker)
    server.STATE.config = make_temp_config(tmp_path)
    server.STATE.tools = [tool_definition()]
    logs = []

    async def fake_call_qwen(messages):
        return {"choices": [{"message": {"content": None, "tool_calls": [qwen_tool_call()]}}]}

    monkeypatch.setattr(server, "call_qwen", fake_call_qwen)
    monkeypatch.setattr(server, "log_request", lambda **kwargs: logs.append(kwargs))

    response = await asgi_request(
        "POST",
        "/v1/chat/completions",
        headers={"x-openclaw-session-key": "delegated-first"},
        json={"messages": [{"role": "user", "content": "run custom"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["function"]["name"] == "custom_tool"
    assert tool_call["function"]["arguments"] == '{"x":1}'
    assert server._SESSION_PENDING_DELEGATED_TOOL_IDS["delegated-first"] == {"call_1"}
    assert logs[-1]["status"] == "delegated_tool_call"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_chat_delegated_first_turn_streams_tool_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_stub_script(tmp_path)
    server.STATE.config = make_temp_config(tmp_path)
    server.STATE.tools = [tool_definition()]

    async def fake_call_qwen(messages):
        return {"choices": [{"message": {"content": "", "tool_calls": [qwen_tool_call()]}}]}

    monkeypatch.setattr(server, "call_qwen", fake_call_qwen)
    monkeypatch.setattr(server, "log_request", lambda **kwargs: None)

    response = await asgi_request(
        "POST",
        "/v1/chat/completions",
        headers={"x-openclaw-session-key": "delegated-stream"},
        json={"stream": True, "messages": [{"role": "user", "content": "run custom"}]},
    )

    assert response.status_code == 200
    chunks = sse_payloads(response.text)
    first_choice = chunks[0]["choices"][0]
    final_choice = chunks[-1]["choices"][0]
    assert first_choice["delta"]["tool_calls"][0]["function"]["arguments"] == '{"x":1}'
    assert final_choice["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_chat_delegated_continuation_resumes_to_qwen_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server.STATE.config = make_temp_config(tmp_path)
    server.STATE.tools = [tool_definition()]
    server._SESSION_PENDING_DELEGATED_TOOL_IDS["resume-ok"] = {"call_1"}
    logs = []

    async def fake_call_qwen(messages):
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["name"] == "custom_tool"
        return {"choices": [{"message": {"content": "完成了"}}]}

    async def fake_completion_check(messages, out_llm_calls=None):
        assert any(message.get("role") == "tool" for message in messages)
        return True

    monkeypatch.setattr(server, "call_qwen", fake_call_qwen)
    monkeypatch.setattr(server, "call_qwen_completion_check", fake_completion_check)
    monkeypatch.setattr(server, "log_request", lambda **kwargs: logs.append(kwargs))

    response = await asgi_request(
        "POST",
        "/v1/chat/completions",
        headers={"x-openclaw-session-key": "resume-ok"},
        json={
            "messages": [
                {"role": "user", "content": "run custom"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [qwen_tool_call(arguments='{"x":1}')],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
            ]
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "完成了"
    assert choice["finish_reason"] == "stop"
    assert logs[-1]["status"] == "qwen_completed"
    assert "resume-ok" in server._QWEN_SAVED_CONTEXTS
    assert "resume-ok" not in server._SESSION_PENDING_DELEGATED_TOOL_IDS
    # OpenClaw already persists the delegated turn natively; queueing a pending
    # plain-text turn here would duplicate it in future upstream requests.
    assert server._QWEN_PENDING_UPSTREAM_TURNS.get("resume-ok") in (None, [])


@pytest.mark.asyncio
async def test_chat_delegated_error_continuation_falls_through_to_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server.STATE.config = make_temp_config(tmp_path)
    server.STATE.tools = [tool_definition()]
    server._SESSION_PENDING_DELEGATED_TOOL_IDS["resume-error"] = {"call_1"}
    logs = []

    async def fake_call_qwen(messages):
        return {"choices": [{"message": {"content": "工具失败"}}]}

    async def fake_completion_check(messages, out_llm_calls=None):
        return False

    async def fake_proxy_upstream(*args, **kwargs):
        return server._build_completion_response("upstream handled")

    monkeypatch.setattr(server, "call_qwen", fake_call_qwen)
    monkeypatch.setattr(server, "call_qwen_completion_check", fake_completion_check)
    monkeypatch.setattr(server, "proxy_upstream", fake_proxy_upstream)
    monkeypatch.setattr(server, "log_request", lambda **kwargs: logs.append(kwargs))

    response = await asgi_request(
        "POST",
        "/v1/chat/completions",
        headers={"x-openclaw-session-key": "resume-error"},
        json={
            "messages": [
                {"role": "user", "content": "run custom"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [qwen_tool_call(arguments='{"x":1}')],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"error":"boom"}'},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "upstream handled"
    assert logs[-1]["status"] == "tool_result_to_upstream"


@pytest.mark.asyncio
async def test_trailing_tool_without_pending_id_keeps_existing_continuation_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server.STATE.config = make_temp_config(tmp_path)
    server.STATE.tools = [tool_definition()]
    logs = []

    async def fail_call_qwen(messages):
        raise AssertionError("Qwen should not be called for upstream-native continuations")

    async def fake_proxy_upstream(*args, **kwargs):
        return server._build_completion_response("upstream continuation")

    monkeypatch.setattr(server, "call_qwen", fail_call_qwen)
    monkeypatch.setattr(server, "proxy_upstream", fake_proxy_upstream)
    monkeypatch.setattr(server, "log_request", lambda **kwargs: logs.append(kwargs))

    response = await asgi_request(
        "POST",
        "/v1/chat/completions",
        headers={"x-openclaw-session-key": "no-pending"},
        json={
            "messages": [
                {"role": "user", "content": "run custom"},
                {"role": "assistant", "content": "", "tool_calls": [qwen_tool_call()]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "upstream continuation"
    assert logs[-1]["status"] == "skipped_qwen_continuation"


@pytest.mark.asyncio
async def test_delegation_disabled_executes_tool_inside_router(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config()
    config.delegate_to_openclaw = False
    server.STATE.config = config
    server.STATE.tools = [tool_definition()]
    calls = [
        {"choices": [{"message": {"content": "", "tool_calls": [qwen_tool_call(arguments='{"x":1}')]}}]},
        {"choices": [{"message": {"content": "done"}}]},
    ]
    executed = []

    async def fake_call_qwen(messages):
        return calls.pop(0)

    async def fake_execute_tool(function_name: str, arguments_json: str):
        executed.append((function_name, arguments_json))
        return {"ok": True}

    monkeypatch.setattr(server, "call_qwen", fake_call_qwen)
    monkeypatch.setattr(server, "execute_tool", fake_execute_tool)

    result = await server.run_tool_loop(
        "run custom",
        delegated_tool_names=(server._delegated_tool_names() or None),
    )

    assert server._delegated_tool_names() == set()
    assert executed == [("custom_tool", '{"x":1}')]
    assert result.delegated_tool_calls == []
    assert result.used_any_tool is True


@pytest.mark.asyncio
async def test_tools_and_execute_tool_endpoints(tmp_path: Path) -> None:
    write_stub_script(tmp_path)
    server.STATE.config = make_temp_config(tmp_path)
    server.STATE.tools = [tool_definition()]

    tools_response = await asgi_request("GET", "/v1/tools")
    assert tools_response.status_code == 200
    assert tools_response.json()["tools"] == [tool_definition()]

    execute_response = await asgi_request(
        "POST",
        "/v1/execute_tool",
        json={"name": "custom_tool", "arguments": {"x": 1}},
    )
    assert execute_response.status_code == 200
    assert execute_response.json() == {"name": "custom_tool", "result": {"ok": True}}

    invalid_response = await asgi_request(
        "POST",
        "/v1/execute_tool",
        json={"name": "", "arguments": {}},
    )
    assert invalid_response.status_code == 400

    unknown_response = await asgi_request(
        "POST",
        "/v1/execute_tool",
        json={"name": "unknown_tool", "arguments": {}},
    )
    assert unknown_response.status_code == 200
    assert unknown_response.json()["result"]["error"] == "script not found: unknown_tool.sh"


@pytest.mark.asyncio
async def test_startup_writes_openclaw_tools_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functions_path = tmp_path / "functions.jsonl"
    functions_path.write_text(
        '{"name":"custom_tool","parameters":{"type":"object","properties":{}}}\n',
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "listen_host": "127.0.0.1",
                "listen_port": 0,
                "routing": {"base_url": "http://router/v1", "model": "router", "api_key": "any"},
                "upstream": {"base_url": "http://upstream/v1", "model": "upstream", "api_key": "any"},
                "functions_file": "functions.jsonl",
                "scripts_dir": "scripts",
                "max_tool_rounds": 1,
                "tool_exec_timeout_s": 5,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "scripts").mkdir()
    server.STATE.config_path = tmp_path / "config.json"

    async def fake_build_http_client():
        return object()

    async def fake_warmup_qwen():
        return False

    monkeypatch.setattr(server, "build_http_client", fake_build_http_client)
    monkeypatch.setattr(server, "warmup_qwen", fake_warmup_qwen)

    await server.startup_event()

    snapshot = json.loads((tmp_path / "openclaw-tools.json").read_text(encoding="utf-8"))
    names = {tool["function"]["name"] for tool in snapshot["tools"]}
    assert "custom_tool" in names
    assert "find" in names
