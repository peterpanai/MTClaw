import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import plugin from "../dist/index.js";

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "fr-tools-"));
fs.writeFileSync(
  path.join(tmpRoot, "config.json"),
  JSON.stringify({ listen_port: 18790 }, null, 2),
  "utf8",
);
fs.writeFileSync(
  path.join(tmpRoot, "openclaw-tools.json"),
  JSON.stringify(
    {
      tools: [
        {
          type: "function",
          function: {
            name: "alpha",
            description: "Alpha tool",
            parameters: { type: "object", properties: { value: { type: "number" } } },
          },
        },
        {
          type: "function",
          function: {
            name: "beta",
            parameters: { type: "object", properties: {} },
          },
        },
      ],
    },
    null,
    2,
  ),
  "utf8",
);

const server = http.createServer((req, res) => {
  if (req.method !== "POST" || req.url !== "/v1/execute_tool") {
    res.writeHead(404, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "not found" }));
    return;
  }
  let body = "";
  req.setEncoding("utf8");
  req.on("data", (chunk) => {
    body += chunk;
  });
  req.on("end", () => {
    const payload = JSON.parse(body || "{}");
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        result: {
          ok: true,
          name: payload.name,
          arguments: payload.arguments,
          message: "stub result",
        },
      }),
    );
  });
});

const listenResult = await new Promise((resolve) => {
  server.once("error", (error) => resolve({ ok: false, error }));
  server.listen(0, "127.0.0.1", () => resolve({ ok: true }));
});

let routerUrl;
let restoreFetch = null;
if (listenResult.ok) {
  const address = server.address();
  routerUrl = `http://127.0.0.1:${address.port}`;
} else if (listenResult.error?.code === "EPERM") {
  const originalFetch = globalThis.fetch;
  restoreFetch = () => {
    globalThis.fetch = originalFetch;
  };
  globalThis.fetch = async (_url, options) => {
    const payload = JSON.parse(options?.body || "{}");
    return new Response(
      JSON.stringify({
        result: {
          ok: true,
          name: payload.name,
          arguments: payload.arguments,
          message: "stub result",
        },
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  };
  routerUrl = "http://127.0.0.1:1";
} else {
  throw listenResult.error;
}

try {
  const registered = [];
  const api = {
    // Same field the OpenClaw loader passes plugins.entries[...].config through.
    pluginConfig: {
      rootDir: tmpRoot,
      routerUrl,
      execTimeoutMs: 5000,
    },
    logger: {
      info() {},
      warn(message) {
        throw new Error(message);
      },
    },
    registerTool(tool) {
      registered.push(tool);
    },
  };

  plugin.register(api);
  assert.deepEqual(
    registered.map((tool) => tool.name),
    ["alpha", "beta"],
  );

  const result = await registered[0].execute(
    "call_alpha",
    { value: 42 },
    new AbortController().signal,
  );
  assert.equal(result.details.ok, true);
  assert.equal(result.details.name, "alpha");
  assert.equal(result.details.message, "stub result");
  assert.match(result.content[0].text, /stub result/);
  assert.match(result.content[0].text, /42/);
} finally {
  if (listenResult.ok) {
    await new Promise((resolve) => server.close(resolve));
  }
  if (restoreFetch) restoreFetch();
  fs.rmSync(tmpRoot, { recursive: true, force: true });
}
