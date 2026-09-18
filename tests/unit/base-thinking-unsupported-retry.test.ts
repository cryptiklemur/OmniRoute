/**
 * TDD for #13868 — BaseExecutor's 400-recovery chain must retry once, stripping the
 * request's reasoning field, when an upstream rejects thinking by naming the MODEL
 * rather than the offending field (Ollama does this for Instruct-only models such as
 * Qwen3-Coder). Neither `findOffendingField` nor `detectUnsupportedParam` can see
 * anything to strip in that error shape, so this exercises the dedicated
 * `isUnsupportedThinkingError` branch end-to-end through the real retry chain.
 *
 * Mirrors the fetch-capture pattern in context-editing-relays.test.ts.
 *
 * Run: node --import tsx/esm --test tests/unit/base-thinking-unsupported-retry.test.ts
 */
import test from "node:test";
import assert from "node:assert/strict";

import { DefaultExecutor } from "../../open-sse/executors/default.ts";

/** First call returns `status` with `errorText`; subsequent calls return 200 OK. */
function mockFetchErrorThenOk(status: number, errorText: string) {
  const bodies: Array<Record<string, unknown>> = [];
  const original = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = (async (_url: unknown, init: { body?: unknown } = {}) => {
    bodies.push(JSON.parse(String(init.body ?? "{}")));
    calls += 1;
    if (calls === 1) {
      return new Response(errorText, {
        status,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof globalThis.fetch;
  return { bodies, callCount: () => calls, restore: () => void (globalThis.fetch = original) };
}

const baseCredentials = { apiKey: "relay-key", baseUrl: "https://relay.example/v1" };

test("#13868: upstream 400 naming the model (not the field) as thinking-unsupported strips reasoning_effort and retries once", async () => {
  const { bodies, callCount, restore } = mockFetchErrorThenOk(
    400,
    '"Qwen3-Coder:latest" does not support thinking'
  );
  try {
    await new DefaultExecutor("anthropic-compatible-cc-myrelay").execute({
      model: "Qwen3-Coder:latest",
      body: {
        model: "Qwen3-Coder:latest",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 1,
        reasoning_effort: "high",
      },
      stream: false,
      credentials: baseCredentials,
    });
  } finally {
    restore();
  }
  assert.equal(callCount(), 2, "must retry exactly once after the thinking-unsupported 400");
  assert.equal(bodies[0]?.reasoning_effort, "high", "first attempt carried reasoning_effort");
  assert.equal(
    "reasoning_effort" in (bodies[1] ?? {}),
    false,
    "retry must drop reasoning_effort entirely"
  );
});

test("#13868: the retry strips only the reasoning field actually present, not every REASONING_REQUEST_FIELDS entry", async () => {
  const { bodies, callCount, restore } = mockFetchErrorThenOk(
    400,
    "model gemma3 does not support reasoning"
  );
  try {
    await new DefaultExecutor("anthropic-compatible-cc-myrelay").execute({
      model: "gemma3",
      body: {
        model: "gemma3",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 1,
        thinking: { type: "enabled" },
      },
      stream: false,
      credentials: baseCredentials,
    });
  } finally {
    restore();
  }
  assert.equal(callCount(), 2, "must retry exactly once");
  assert.equal("thinking" in (bodies[1] ?? {}), false, "retry must drop the present field");
  assert.equal(
    "reasoning_effort" in (bodies[1] ?? {}),
    false,
    "a field never sent must not appear on retry either"
  );
});

test("#13868: an UNRELATED 400 does NOT strip reasoning fields or retry", async () => {
  const { bodies, callCount, restore } = mockFetchErrorThenOk(400, "max_tokens: must be >= 1");
  try {
    await new DefaultExecutor("anthropic-compatible-cc-myrelay").execute({
      model: "Qwen3-Coder:latest",
      body: {
        model: "Qwen3-Coder:latest",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 1,
        reasoning_effort: "high",
      },
      stream: false,
      credentials: baseCredentials,
    });
  } finally {
    restore();
  }
  assert.equal(callCount(), 1, "an unrelated 400 must not trigger the thinking-unsupported retry");
  assert.equal(bodies[0]?.reasoning_effort, "high", "the single attempt still carried the field");
});
