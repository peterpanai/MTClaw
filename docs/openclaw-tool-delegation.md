# OpenClaw Tool Delegation

Function Router delegates tool execution to OpenClaw by default so the upstream model sees real OpenClaw session history instead of a hidden router-side tool trace.

## Flow

```text
User -> OpenClaw -> Function Router
Function Router routing model returns a tool call
Function Router -> OpenClaw: assistant.tool_calls, finish_reason=tool_calls
OpenClaw fr-tools plugin executes the tool via /v1/execute_tool
OpenClaw stores assistant.tool_calls + role=tool in the session
OpenClaw -> Function Router: continuation ending in role=tool
Function Router resumes completion check, then returns the router reply or falls through upstream
```

## Why This Fixes Hallucinated Completion

Without delegation, Function Router executes tools internally and forwards to upstream without real OpenClaw tool messages. The upstream model can then claim work is done without seeing the tool call/result, or ask for tools OpenClaw never registered.

With delegation, OpenClaw owns the visible tool call and result. The continuation back to Function Router includes the actual `assistant.tool_calls` and `role=tool` messages, so the session history remains coherent.

Because OpenClaw persists the whole delegated turn natively (user message, `assistant.tool_calls`, `role=tool` result, and the final reply), Function Router does not additionally queue these turns into its pending plain-text upstream history — future upstream requests see exactly one structured copy of each tool interaction, which upstream models can also reuse to issue the same tools themselves.

## fr-tools Plugin

`plugins/fr-tools/` registers every Function Router tool as an OpenClaw executable tool. Registration is synchronous: it reads `<rootDir>/openclaw-tools.json`, falling back to `<rootDir>/functions.jsonl` if the snapshot is missing.

At runtime, each registered tool posts to:

```text
POST /v1/execute_tool
{"name":"tool_name","arguments":{...}}
```

Function Router keeps the existing execution path for wrapper scripts, builtins, `FR_TOOLS_BASE_DIR`, and timeouts.

## Snapshot And Endpoints

- `openclaw-tools.json`: written best-effort on Function Router startup as `{"tools":[...]}`.
- `GET /v1/tools`: returns the currently loaded tool definitions.
- `POST /v1/execute_tool`: executes one tool and returns `{"name": "...", "result": ...}`.

Disable delegation for clients that cannot execute OpenAI tool calls:

```json
{
  "delegate_tools_to_openclaw": false
}
```
