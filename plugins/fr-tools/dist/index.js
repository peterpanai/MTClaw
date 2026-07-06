import fs from "node:fs";
import os from "node:os";
import path from "node:path";
function pluginConfig(api) {
    let fromGetter = {};
    try {
        if (typeof api?.getConfig === "function") {
            fromGetter = api.getConfig("fr-tools") || api.getConfig() || {};
        }
    }
    catch {
        fromGetter = {};
    }
    // api.pluginConfig is what the OpenClaw loader actually passes for
    // plugins.entries["fr-tools"].config — it must win over the fallbacks.
    return {
        ...(api?.config || {}),
        ...(api?.settings || {}),
        ...(fromGetter || {}),
        ...(api?.pluginConfig && typeof api.pluginConfig === "object" && !Array.isArray(api.pluginConfig)
            ? api.pluginConfig
            : {}),
    };
}
function expandHome(value) {
    if (!value || value === "~")
        return os.homedir();
    if (value.startsWith("~/"))
        return path.join(os.homedir(), value.slice(2));
    return value;
}
function resolveRootDir(config) {
    const configured = typeof config?.rootDir === "string" ? config.rootDir.trim() : "";
    return path.resolve(expandHome(configured || path.join(os.homedir(), ".function-router")));
}
function readListenPort(rootDir) {
    try {
        const raw = fs.readFileSync(path.join(rootDir, "config.json"), "utf8");
        const parsed = JSON.parse(raw);
        const port = Number(parsed?.listen_port);
        return Number.isFinite(port) && port > 0 ? port : 18790;
    }
    catch {
        return 18790;
    }
}
function resolveRouterUrl(config, rootDir) {
    const configured = typeof config?.routerUrl === "string" ? config.routerUrl.trim() : "";
    if (configured)
        return configured.replace(/\/+$/, "");
    return `http://127.0.0.1:${readListenPort(rootDir)}`;
}
function normalizeTools(payload) {
    if (Array.isArray(payload?.tools))
        return payload.tools;
    if (Array.isArray(payload))
        return payload;
    return [];
}
function loadSnapshot(rootDir) {
    const snapshotPath = path.join(rootDir, "openclaw-tools.json");
    const raw = fs.readFileSync(snapshotPath, "utf8");
    return normalizeTools(JSON.parse(raw));
}
function loadFunctionsJsonl(rootDir) {
    const functionsPath = path.join(rootDir, "functions.jsonl");
    const raw = fs.readFileSync(functionsPath, "utf8");
    const tools = [];
    for (const line of raw.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed)
            continue;
        tools.push({ type: "function", function: JSON.parse(trimmed) });
    }
    return tools;
}
function loadTools(rootDir, logger) {
    try {
        return loadSnapshot(rootDir) || [];
    }
    catch (snapshotError) {
        try {
            return loadFunctionsJsonl(rootDir) || [];
        }
        catch (functionsError) {
            logger.warn?.(`[fr-tools] no readable tool snapshot or functions.jsonl under ${rootDir}`);
            return [];
        }
    }
}
function resultPayload(payload) {
    return {
        content: [
            {
                type: "text",
                text: JSON.stringify(payload, null, 2),
            },
        ],
        details: payload,
    };
}
function errorPayload(error) {
    return resultPayload({ ok: false, error });
}
function errorMessage(error) {
    if (error?.name === "AbortError")
        return "tool execution aborted or timed out";
    if (typeof error?.message === "string" && error.message)
        return error.message;
    return String(error || "tool execution failed");
}
async function executeRouterTool(routerUrl, name, params, signal, execTimeoutMs) {
    const controller = new AbortController();
    const timeout = setTimeout(() => {
        controller.abort(new Error(`tool execution timed out after ${execTimeoutMs}ms`));
    }, execTimeoutMs);
    const abortFromCaller = () => controller.abort(signal?.reason || new Error("tool execution aborted"));
    if (signal?.addEventListener) {
        if (signal.aborted)
            abortFromCaller();
        else
            signal.addEventListener("abort", abortFromCaller, { once: true });
    }
    try {
        const response = await fetch(`${routerUrl}/v1/execute_tool`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, arguments: params || {} }),
            signal: controller.signal,
        });
        const rawText = await response.text();
        let data;
        try {
            data = rawText ? JSON.parse(rawText) : {};
        }
        catch {
            data = { ok: false, error: rawText || `HTTP ${response.status}` };
        }
        if (!response.ok) {
            const detail = data?.detail || data?.error || rawText || response.statusText;
            return errorPayload(`HTTP ${response.status}: ${detail}`);
        }
        return resultPayload(data.result ?? data);
    }
    catch (error) {
        return errorPayload(errorMessage(error));
    }
    finally {
        clearTimeout(timeout);
        if (signal?.removeEventListener) {
            signal.removeEventListener("abort", abortFromCaller);
        }
    }
}
const frToolsPlugin = {
    id: "fr-tools",
    name: "Function Router Tools",
    description: "Registers Function Router tools for OpenClaw-side execution",
    register(api) {
        const logger = api?.logger || console;
        const config = pluginConfig(api);
        const rootDir = resolveRootDir(config);
        const routerUrl = resolveRouterUrl(config, rootDir);
        const execTimeoutMs = Number(config?.execTimeoutMs || 300000);
        const tools = loadTools(rootDir, logger);
        const registeredNames = [];
        for (const tool of tools) {
            const fn = tool?.function || {};
            const name = typeof fn?.name === "string" ? fn.name : "";
            if (!name) {
                logger.warn?.("[fr-tools] skipping tool without function.name");
                continue;
            }
            try {
                api.registerTool({
                    name,
                    label: name,
                    description: fn.description || name,
                    parameters: fn.parameters || { type: "object", properties: {} },
                    async execute(_toolCallId, params, signal) {
                        return executeRouterTool(routerUrl, name, params, signal, execTimeoutMs);
                    },
                });
                registeredNames.push(name);
            }
            catch (error) {
                logger.warn?.(`[fr-tools] skipping ${name}: ${errorMessage(error)}`);
            }
        }
        logger.info?.(`[fr-tools] registered tools: ${registeredNames.join(", ") || "(none)"}`);
    },
};
export default frToolsPlugin;
