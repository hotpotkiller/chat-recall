// Offline checks: node browser-extension/test_adapter.mjs
// These fixtures do not validate live ChatGPT endpoints or browser permissions.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import crypto from "node:crypto";
import "./adapter.js";

const adapter = globalThis.ChatRecallAdapter;
assert.equal(adapter.endpoint("http://127.0.0.1:8765"), "http://127.0.0.1:8765");
for (const value of ["https://example.com", "http://example.com", "http://127.0.0.1.evil.test", "http://localhost/path", "http://user:pass@localhost", "http://localhost/?token=secret"]) assert.throws(() => adapter.endpoint(value));
assert.equal(adapter.version({ update_time: 100 }), "100000");
assert.equal(adapter.version({ update_time: "1970-01-01T00:01:40Z" }), "100000");
assert.equal(adapter.version({ update_time: "not-a-date" }), null);
assert.throws(() => adapter.page({ error: "new format" }, 0, 20));
assert.throws(() => adapter.page({ items: [{}] }, 0, 20));
assert.equal(adapter.page({ items: [{ id: "a" }], total: 1 }, 0, 20).exhausted, true);
assert.equal(adapter.page({ items: [{ id: "a" }], cursor: "next", has_more: true }, 0, 20).exhausted, false);
assert.equal(adapter.page({ items: [], has_missing_conversations: true }, 20, 20).missing, true);
assert.equal(adapter.checkpoint({ offset: -1 }).offset, 0);
assert.equal(adapter.checkpoint({ offset: 20, phase: "backfill" }).offset, 20);
console.log("Browser adapter offline checks passed (endpoint, time, schema, pagination, checkpoint).");

// --- Synthetic VM execution of actual background.js ---
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const backgroundCode = fs.readFileSync(path.join(__dirname, "background.js"), "utf8");
const adapterCode = fs.readFileSync(path.join(__dirname, "adapter.js"), "utf8");

function mockResponse({ status = 200, headers = {}, json = {}, malformedJson = false }) {
  const headerMap = new Map();
  for (const [k, v] of Object.entries(headers)) {
    headerMap.set(k.toLowerCase(), String(v));
  }
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: {
      get: (name) => headerMap.get(name.toLowerCase()) || null,
      ...headers
    },
    json: async () => {
      if (malformedJson) throw new SyntaxError("Unexpected token < in JSON at position 0");
      return json;
    },
    text: async () => JSON.stringify(json)
  };
}

function createSyntheticJwt({ exp, extraClaims = {} } = {}) {
  const header = Buffer.from(JSON.stringify({ alg: "none", typ: "JWT" })).toString("base64url");
  const payloadObj = { ...(exp !== undefined ? { exp } : {}), ...extraClaims };
  const payload = Buffer.from(JSON.stringify(payloadObj)).toString("base64url");
  const sig = Buffer.from("synthetic-sig").toString("base64url");
  return `${header}.${payload}.${sig}`;
}

function createBootstrapElement({
  tagName = "SCRIPT",
  type = "application/json",
  session = null,
  rawText = null
} = {}) {
  const textContent = rawText !== null ? rawText : JSON.stringify({ session });
  return {
    tagName,
    nodeName: tagName,
    type,
    getAttribute(name) {
      if (typeof name === "string" && name.toLowerCase() === "type") return type;
      return null;
    },
    textContent
  };
}

function createMockDocument({ element = null } = {}) {
  return {
    getElementById(id) {
      if (id === "client-bootstrap") return element;
      return null;
    }
  };
}

function cloneDeep(val) {
  if (val === undefined) return undefined;
  return JSON.parse(JSON.stringify(val));
}

function createSyntheticEnv({
  fetchHandler,
  storage = {},
  store = null,
  tabs = [{ id: 1, active: true, discarded: false, url: "https://chatgpt.com/" }],
  document = undefined,
  initialTime = Date.now(),
  random = 0.1
}) {
  const effectiveStore = store instanceof Map
    ? store
    : new Map(Object.entries(storage).map(([k, v]) => [k, cloneDeep(v)]));
  const storageLocal = {
    async get(keys) {
      if (typeof keys === "string") return { [keys]: cloneDeep(effectiveStore.get(keys)) };
      if (Array.isArray(keys)) {
        const out = {};
        for (const k of keys) out[k] = cloneDeep(effectiveStore.get(k));
        return out;
      }
      if (keys && typeof keys === "object") {
        const out = { ...keys };
        for (const k of Object.keys(keys)) {
          if (effectiveStore.has(k)) out[k] = cloneDeep(effectiveStore.get(k));
        }
        return out;
      }
      const out = {};
      for (const [k, v] of effectiveStore.entries()) out[k] = cloneDeep(v);
      return out;
    },
    async set(items) {
      for (const [k, v] of Object.entries(items)) effectiveStore.set(k, cloneDeep(v));
    },
    async remove(keys) {
      const arr = Array.isArray(keys) ? keys : [keys];
      for (const k of arr) effectiveStore.delete(k);
    }
  };

  const alarms = new Map();
  const alarmListeners = [];
  const startupListeners = [];
  const messageListeners = [];

  let currentTime = initialTime;

  const chromeMock = {
    runtime: {
      id: "synthetic-test-extension",
      onMessage: {
        addListener: (fn) => messageListeners.push(fn)
      },
      onStartup: {
        addListener: (fn) => startupListeners.push(fn)
      }
    },
    alarms: {
      clear: async (name) => {
        alarms.delete(name);
      },
      create: async (name, info) => {
        const scheduledTime = info.when !== undefined ? info.when : (currentTime + (info.delayInMinutes || 0) * 60000);
        alarms.set(name, { name, scheduledTime, ...info });
      },
      get: async (name) => {
        return alarms.get(name) || null;
      },
      onAlarm: {
        addListener: (fn) => alarmListeners.push(fn)
      }
    },
    storage: { local: storageLocal },
    tabs: { query: async () => tabs },
    scripting: {
      executeScript: async ({ target, world, func, args }) => {
        const result = await func(...args);
        return [{ result }];
      }
    }
  };

  const webCrypto = globalThis.crypto?.subtle ? globalThis.crypto : crypto.webcrypto;

  class FakeDate extends Date {
    constructor(...args) {
      if (args.length === 0) super(currentTime);
      else super(...args);
    }
    static now() {
      return currentTime;
    }
  }
  FakeDate.UTC = Date.UTC;
  FakeDate.parse = Date.parse;

  const fakeMath = Object.create(Math);
  let currentRandom = random;
  fakeMath.random = () => (typeof currentRandom === "function" ? currentRandom() : currentRandom);

  const sandbox = {
    console,
    setTimeout: (fn, ms) => setTimeout(fn, Math.min(ms, 1)),
    clearTimeout,
    Promise,
    TextEncoder,
    Uint8Array,
    Array,
    Object,
    String,
    Number,
    Date: FakeDate,
    Math: fakeMath,
    URL,
    crypto: webCrypto,
    location: { origin: "https://chatgpt.com" },
    chrome: chromeMock,
    document,
    atob: globalThis.atob || ((b64) => Buffer.from(b64, "base64").toString("binary")),
    fetch: (url, opts) => fetchHandler(String(url), opts),
    importScripts: (file) => {
      if (file === "adapter.js") {
        vm.runInContext(adapterCode, context);
      }
    }
  };

  const context = vm.createContext(sandbox);
  vm.runInContext(backgroundCode, context);

  return {
    context,
    store: effectiveStore,
    readPage: context.readPage,
    readChatGPT: context.readChatGPT,
    syncHistory: context.syncHistory,
    handleMessage: context.handleMessage,
    currentTime: () => currentTime,
    advanceTime: (ms) => { currentTime += ms; },
    setTime: (t) => { currentTime = t; },
    setRandom: (r) => { currentRandom = r; },
    alarms,
    fireAlarm: async (name = "chat-recall-sync") => {
      const alarm = alarms.get(name);
      if (alarm) {
        for (const fn of alarmListeners) await fn(alarm);
      }
    },
    fireStartup: async () => {
      for (const fn of startupListeners) await fn();
    },
    sendMessage: async (message) => {
      return context.handleMessage(message);
    }
  };
}

async function runSyntheticTests() {
  // Test 1: Session 401 vs 403 vs challenge vs 429
  {
    // 401
    const env401 = createSyntheticEnv({
      fetchHandler: async () => mockResponse({ status: 401 })
    });
    const res401 = await env401.readPage({ operation: "identity" });
    assert.equal(res401.status, 401);
    assert.equal(res401.stage, "session");
    assert.match(res401.error, /ChatGPT session returned 401/);
    assert.match(res401.error, /Sign in again/);

    // 403 (standard)
    const env403 = createSyntheticEnv({
      fetchHandler: async () => mockResponse({ status: 403 })
    });
    const res403 = await env403.readPage({ operation: "identity" });
    assert.equal(res403.status, 403);
    assert.equal(res403.stage, "session");
    assert.equal(res403.challenge, undefined);
    assert.match(res403.error, /ChatGPT session returned 403/);
    assert.match(res403.error, /request refused/i);
    assert.match(res403.error, /account may still be signed in/i);
    assert.match(res403.error, /retry manually/i);

    // 403 (challenge)
    const envChallenge = createSyntheticEnv({
      fetchHandler: async () => mockResponse({ status: 403, headers: { "cf-mitigated": "challenge" } })
    });
    const resChallenge = await envChallenge.readPage({ operation: "identity" });
    assert.equal(resChallenge.status, 403);
    assert.equal(resChallenge.stage, "session");
    assert.equal(resChallenge.challenge, true);
    assert.match(resChallenge.error, /challenge/i);
    assert.match(resChallenge.error, /request refused/i);
    assert.match(resChallenge.error, /retry manually/i);

    // 429
    const env429 = createSyntheticEnv({
      fetchHandler: async () => mockResponse({ status: 429 })
    });
    const res429 = await env429.readPage({ operation: "identity" });
    assert.equal(res429.status, 429);
    assert.equal(res429.stage, "session");
    assert.match(res429.error, /ChatGPT session returned 429/);
    assert.match(res429.error, /Rate limited/i);
    assert.match(res429.error, /Wait before retrying/i);
  }

  // Test 2: Malformed JSON across session, history, detail
  {
    const envSessionJson = createSyntheticEnv({
      fetchHandler: async () => mockResponse({ status: 200, malformedJson: true })
    });
    const resSessionJson = await envSessionJson.readPage({ operation: "identity" });
    assert.equal(resSessionJson.status, 200);
    assert.equal(resSessionJson.stage, "session");
    assert.equal(resSessionJson.error, "ChatGPT session returned unreadable response data. The checkpoint is preserved.");
    assert.doesNotMatch(resSessionJson.error, /SyntaxError|<|position/);

    const envHistoryJson = createSyntheticEnv({
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: "tok-abc", user: { id: "user-123" } } });
        }
        return mockResponse({ status: 200, malformedJson: true });
      }
    });
    const resHistoryJson = await envHistoryJson.readPage({ operation: "page", limit: 20, offset: 0 });
    assert.equal(resHistoryJson.status, 200);
    assert.equal(resHistoryJson.stage, "history");
    assert.equal(resHistoryJson.error, "ChatGPT history returned unreadable response data. The checkpoint is preserved.");
    assert.doesNotMatch(resHistoryJson.error, /SyntaxError|<|position/);

    const envDetailJson = createSyntheticEnv({
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: "tok-abc", user: { id: "user-123" } } });
        }
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({ status: 200, json: { items: [{ id: "secret-conv-id-777", update_time: 1000 }] } });
        }
        return mockResponse({ status: 200, malformedJson: true });
      }
    });
    const resDetailJson = await envDetailJson.readPage({ operation: "page", limit: 20, offset: 0 });
    assert.equal(resDetailJson.status, 200);
    assert.equal(resDetailJson.stage, "conversation");
    assert.equal(resDetailJson.error, "ChatGPT conversation returned unreadable response data. The checkpoint is preserved.");
    assert.doesNotMatch(resDetailJson.error, /SyntaxError|<|position/);
    assert.equal(resDetailJson.error.includes("secret-conv-id-777"), false);
  }

  // Test 3: History and detail HTTP failures & zero conversation ID leakage
  {
    const envHist403 = createSyntheticEnv({
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: "tok-abc", user: { id: "user-123" } } });
        }
        return mockResponse({ status: 403, headers: { "cf-mitigated": "challenge" } });
      }
    });
    const resHist403 = await envHist403.readPage({ operation: "page", limit: 20, offset: 0 });
    assert.equal(resHist403.status, 403);
    assert.equal(resHist403.stage, "history");
    assert.equal(resHist403.challenge, true);
    assert.match(resHist403.error, /ChatGPT history returned 403/);

    const envDetailFail = createSyntheticEnv({
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: "tok-abc", user: { id: "user-123" } } });
        }
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({ status: 200, json: { items: [{ id: "private-chat-guid-999", update_time: 1000 }] } });
        }
        return mockResponse({ status: 500 });
      }
    });
    const resDetailFail = await envDetailFail.readPage({ operation: "page", limit: 20, offset: 0 });
    assert.equal(resDetailFail.status, 500);
    assert.equal(resDetailFail.stage, "conversation");
    assert.equal(resDetailFail.error.includes("private-chat-guid-999"), false);
    assert.match(resDetailFail.error, /ChatGPT conversation returned 500/);
  }

  // Test 4: Regressions - plain-object throw and session-stage network failure
  {
    const envPlainThrow = createSyntheticEnv({
      fetchHandler: async () => {
        throw { error: "SYNTHETIC_EXCEPTION_SECRET", secret_data: 999 };
      }
    });
    const resPlainThrow = await envPlainThrow.readPage({ operation: "identity" });
    assert.equal(resPlainThrow.stage, "session");
    assert.equal(resPlainThrow.error, "ChatGPT session request failed. The checkpoint is preserved.");
    assert.equal(resPlainThrow.error.includes("SYNTHETIC_EXCEPTION_SECRET"), false);
    assert.equal(resPlainThrow.secret_data, undefined);

    const envNetError = createSyntheticEnv({
      fetchHandler: async () => {
        throw new TypeError("Failed to fetch");
      }
    });
    const resNetError = await envNetError.readPage({ operation: "identity" });
    assert.equal(resNetError.stage, "session");
    assert.equal(resNetError.error, "ChatGPT session request failed. The checkpoint is preserved.");
    assert.doesNotMatch(resNetError.error, /Failed to fetch/);
  }

  // Test 5: Verify session 403 makes exactly one request per explicit attempt (no hidden retry)
  {
    let sessionRequests = 0;
    const envNoHiddenRetry = createSyntheticEnv({
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "" }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          sessionRequests++;
          return mockResponse({ status: 403, headers: { "cf-mitigated": "challenge" } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const firstRun = await envNoHiddenRetry.syncHistory({ manual: true });
    assert.equal(firstRun.ok, false);
    assert.equal(firstRun.status, 403);
    assert.equal(firstRun.stage, "session");
    assert.equal(firstRun.challenge, true);
    assert.equal(sessionRequests, 1, "Session 403 must trigger exactly one request on initial attempt");

    const secondRun = await envNoHiddenRetry.syncHistory({ manual: true });
    assert.equal(secondRun.ok, false);
    assert.equal(secondRun.status, 403);
    assert.equal(sessionRequests, 2, "Second explicit attempt must trigger exactly one additional request");
  }

  // Test 6: Verify status/result exclude all three synthetic sentinel strings
  {
    const SENTINEL_ACCESS_TOKEN = "SENTINEL_ACCESS_TOKEN_ALPHA_123";
    const SENTINEL_PAIRING_TOKEN = "SENTINEL_PAIRING_TOKEN_BETA_456";
    const SENTINEL_CONV_ID = "SENTINEL_CONV_ID_GAMMA_789";
    const SENTINELS = [SENTINEL_ACCESS_TOKEN, SENTINEL_PAIRING_TOKEN, SENTINEL_CONV_ID];

    const envSentinelLeakCheck = createSyntheticEnv({
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: SENTINEL_PAIRING_TOKEN, workspaceId: "" }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: SENTINEL_ACCESS_TOKEN, user: { id: "user-sentinel-1" } } });
        }
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({ status: 200, json: { items: [{ id: SENTINEL_CONV_ID, update_time: 1000 }] } });
        }
        if (url.includes("/backend-api/conversation/")) {
          return mockResponse({ status: 403, headers: { "cf-mitigated": "challenge" } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const syncRes = await envSentinelLeakCheck.syncHistory({ manual: true });
    assert.equal(syncRes.ok, false);
    assert.equal(syncRes.status, 403);
    assert.equal(syncRes.stage, "conversation");

    const syncResultString = JSON.stringify(syncRes);
    const storageStatusString = JSON.stringify(envSentinelLeakCheck.store.get("status"));

    for (const sentinel of SENTINELS) {
      assert.equal(syncResultString.includes(sentinel), false, `Sync result must exclude sentinel: ${sentinel}`);
      assert.equal(storageStatusString.includes(sentinel), false, `Storage status must exclude sentinel: ${sentinel}`);
    }
  }

  // Test 7: Seed a nonzero checkpoint and verify it is unchanged for failures
  {
    const tempEnv = createSyntheticEnv({
      fetchHandler: async () => mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "user-seeded" } } })
    });
    const identRes = await tempEnv.readPage({ operation: "identity" });
    const seededAccountId = identRes.accountId;
    const seededKey = `checkpoint:${seededAccountId}`;

    const SEEDED_CHECKPOINT = {
      offset: 40,
      cursor: "cursor-seed-page2",
      versions: { "seeded-item-1": "1600000000000" },
      phase: "backfill",
      pages: 2
    };

    // Subtest A: Session failure preserves seeded checkpoint
    {
      const envA = createSyntheticEnv({
        storage: {
          settings: { endpoint: "http://127.0.0.1:8765", token: "pair-tok", workspaceId: "" },
          [seededKey]: { ...SEEDED_CHECKPOINT }
        },
        fetchHandler: async (url) => {
          if (url.includes("/api/auth/session")) return mockResponse({ status: 403, headers: { "cf-mitigated": "challenge" } });
          return mockResponse({ status: 200, json: {} });
        }
      });
      const resA = await envA.syncHistory({ manual: true });
      assert.equal(resA.ok, false);
      assert.equal(resA.stage, "session");
      assert.deepEqual(envA.store.get(seededKey), SEEDED_CHECKPOINT, "Seeded checkpoint must remain unchanged on session failure");
    }

    // Subtest B: History failure preserves seeded checkpoint
    {
      const envB = createSyntheticEnv({
        storage: {
          settings: { endpoint: "http://127.0.0.1:8765", token: "pair-tok", workspaceId: "" },
          [seededKey]: { ...SEEDED_CHECKPOINT }
        },
        fetchHandler: async (url) => {
          if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "user-seeded" } } });
          if (url.includes("/backend-api/conversations")) return mockResponse({ status: 500 });
          return mockResponse({ status: 200, json: {} });
        }
      });
      const resB = await envB.syncHistory({ manual: true });
      assert.equal(resB.ok, false);
      assert.equal(resB.stage, "history");
      assert.deepEqual(envB.store.get(seededKey), SEEDED_CHECKPOINT, "Seeded checkpoint must remain unchanged on history failure");
    }

    // Subtest C: Detail failure preserves seeded checkpoint
    {
      const envC = createSyntheticEnv({
        storage: {
          settings: { endpoint: "http://127.0.0.1:8765", token: "pair-tok", workspaceId: "" },
          [seededKey]: { ...SEEDED_CHECKPOINT }
        },
        fetchHandler: async (url) => {
          if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "user-seeded" } } });
          if (url.includes("/backend-api/conversations")) {
            return mockResponse({ status: 200, json: { items: [{ id: "c-detail-fail", update_time: 2000 }], total: 50 } });
          }
          if (url.includes("/backend-api/conversation/")) return mockResponse({ status: 429 });
          return mockResponse({ status: 200, json: {} });
        }
      });
      const resC = await envC.syncHistory({ manual: true });
      assert.equal(resC.ok, false);
      assert.equal(resC.stage, "conversation");
      assert.deepEqual(envC.store.get(seededKey), SEEDED_CHECKPOINT, "Seeded checkpoint must remain unchanged on detail failure");
    }

    // Subtest D: Local-import failure preserves seeded checkpoint
    {
      const envD = createSyntheticEnv({
        storage: {
          settings: { endpoint: "http://127.0.0.1:8765", token: "pair-tok", workspaceId: "" },
          [seededKey]: { ...SEEDED_CHECKPOINT }
        },
        fetchHandler: async (url) => {
          if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "user-seeded" } } });
          if (url.includes("/backend-api/conversations")) {
            return mockResponse({ status: 200, json: { items: [{ id: "c-imp-fail", update_time: 2000 }], total: 50 } });
          }
          if (url.includes("/backend-api/conversation/")) return mockResponse({ status: 200, json: { mapping: { id: "m1" } } });
          if (url.includes("/api/import")) return mockResponse({ status: 500, json: { error: "Database lock" } });
          return mockResponse({ status: 200, json: {} });
        }
      });
      const resD = await envD.syncHistory({ manual: true });
      assert.equal(resD.ok, false);
      assert.deepEqual(envD.store.get(seededKey), SEEDED_CHECKPOINT, "Seeded checkpoint must remain unchanged on local import failure");
    }
  }

  // Test 8: Explicit retry succeeding and setting checkpoint only after local import success
  {
    let sessionMode = "fail403";
    let importCalled = false;
    let importSuccess = false;

    const envRetry = createSyntheticEnv({
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "pair-tok", workspaceId: "" }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          if (sessionMode === "fail403") {
            return mockResponse({ status: 403, headers: { "cf-mitigated": "challenge" } });
          }
          return mockResponse({ status: 200, json: { accessToken: "access-token-ok", user: { id: "user-456" } } });
        }
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({
            status: 200,
            json: { items: [{ id: "c1", update_time: 1600000000 }], total: 1 }
          });
        }
        if (url.includes("/backend-api/conversation/")) {
          return mockResponse({ status: 200, json: { mapping: { id: "root" } } });
        }
        if (url.includes("/api/import")) {
          importCalled = true;
          if (!importSuccess) {
            return mockResponse({ status: 500, json: { error: "Local database temporarily locked" } });
          }
          return mockResponse({ status: 200, json: { ok: true, conversations: 1, errors: [] } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const attempt1 = await envRetry.syncHistory({ manual: true });
    assert.equal(attempt1.ok, false);
    assert.equal(attempt1.status, 403);
    assert.equal(attempt1.stage, "session");
    assert.equal(attempt1.challenge, true);
    assert.equal(importCalled, false);
    let checkpointKeys = Array.from(envRetry.store.keys()).filter((k) => k.startsWith("checkpoint:"));
    assert.equal(checkpointKeys.length, 0);

    sessionMode = "success";
    importSuccess = false;
    const attempt2 = await envRetry.syncHistory({ manual: true });
    assert.equal(attempt2.ok, false);
    assert.equal(importCalled, true);
    checkpointKeys = Array.from(envRetry.store.keys()).filter((k) => k.startsWith("checkpoint:"));
    assert.equal(checkpointKeys.length, 0);

    importSuccess = true;
    const attempt3 = await envRetry.syncHistory({ manual: true });
    assert.equal(attempt3.ok, true);
    assert.equal(attempt3.imported, 1);
    assert.equal(attempt3.checkpointed, true);

    checkpointKeys = Array.from(envRetry.store.keys()).filter((k) => k.startsWith("checkpoint:"));
    assert.equal(checkpointKeys.length, 1);
    const checkpoint = envRetry.store.get(checkpointKeys[0]);
    assert.equal(checkpoint.phase, "poll");
    assert.equal(checkpoint.pages, 1);
    assert.equal(checkpoint.versions["c1"], "1600000000000");

    const finalStatus = envRetry.store.get("status");
    assert.equal(finalStatus.title, "Ordinary-list scan caught up");
    assert.equal(finalStatus.error, undefined);
  }

  // Test 9: Valid bootstrap identity and page with zero session-endpoint fetches
  {
    const validJwt = createSyntheticJwt({ exp: Math.floor(Date.now() / 1000) + 3600, extraClaims: { sub: "user-bootstrap-1" } });
    const bootstrapEl = createBootstrapElement({
      session: {
        accessToken: validJwt,
        user: { id: "user-bootstrap-1" },
        account: { id: "account-should-be-ignored" }
      }
    });

    let sessionCalls = 0;
    let listCalls = 0;
    let detailCalls = 0;
    let seenAuthHeader = null;

    const env = createSyntheticEnv({
      document: createMockDocument({ element: bootstrapEl }),
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-bootstrap-test", workspaceId: "" }
      },
      fetchHandler: async (url, opts) => {
        if (url.includes("/api/auth/session")) {
          sessionCalls++;
          return mockResponse({ status: 500, json: { error: "Session endpoint must not be called" } });
        }
        if (url.includes("/backend-api/conversations")) {
          listCalls++;
          seenAuthHeader = opts?.headers?.Authorization || opts?.headers?.["Authorization"];
          return mockResponse({
            status: 200,
            json: { items: [{ id: "conv-boot-1", update_time: 1000 }], total: 1 }
          });
        }
        if (url.includes("/backend-api/conversation/")) {
          detailCalls++;
          return mockResponse({ status: 200, json: { mapping: { id: "root" } } });
        }
        if (url.includes("/api/import")) {
          return mockResponse({ status: 200, json: { ok: true, conversations: 1, errors: [] } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const identRes = await env.readPage({ operation: "identity" });
    assert.equal(sessionCalls, 0, "Identity with valid bootstrap must not fetch /api/auth/session");
    assert.equal(identRes.session_source, "page-bootstrap");
    assert.match(identRes.accountId, /^web-[0-9a-f]{24}$/);

    const pageRes = await env.readPage({ operation: "page", limit: 20, offset: 0 });
    assert.equal(sessionCalls, 0, "Page read with valid bootstrap must not fetch /api/auth/session");
    assert.equal(pageRes.session_source, "page-bootstrap");
    assert.equal(listCalls, 1);
    assert.equal(detailCalls, 1);
    assert.equal(seenAuthHeader, `Bearer ${validJwt}`);
    assert.equal(pageRes.conversations.length, 1);
    assert.equal(pageRes.conversations[0].id, "conv-boot-1");

    const syncRes = await env.syncHistory({ manual: true });
    assert.equal(sessionCalls, 0, "syncHistory with valid bootstrap must make 0 session endpoint calls");
    assert.equal(syncRes.ok, true);
    assert.equal(syncRes.session_source, "page-bootstrap");
    assert.equal(syncRes.imported, 1);

    const status = env.store.get("status");
    assert.equal(status.session_source, "page-bootstrap");
    assert.match(status.detail, /Session source: page-bootstrap\./);
  }

  // Test 10: Stable identity equality to endpoint fallback
  {
    const userId = "stable-user-xyz-42";
    const validJwt = createSyntheticJwt({ exp: Math.floor(Date.now() / 1000) + 3600, extraClaims: { sub: userId } });
    const bootstrapEl = createBootstrapElement({
      session: {
        accessToken: validJwt,
        user: { id: userId },
        account: { id: "acc-ignored-42" }
      }
    });

    const envBootstrap = createSyntheticEnv({
      document: createMockDocument({ element: bootstrapEl }),
      fetchHandler: async () => mockResponse({ status: 500 })
    });

    const envEndpoint = createSyntheticEnv({
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: "tok-fallback", user: { id: userId } } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const resBootWs = await envBootstrap.readPage({ operation: "identity", workspaceId: "workspace-beta" });
    const resEndWs = await envEndpoint.readPage({ operation: "identity", workspaceId: "workspace-beta" });
    assert.equal(resBootWs.accountId, resEndWs.accountId, "Account ID must match exactly between bootstrap and endpoint");
    assert.equal(resBootWs.session_source, "page-bootstrap");
    assert.equal(resEndWs.session_source, "session-endpoint");

    const resBootDef = await envBootstrap.readPage({ operation: "identity", workspaceId: "" });
    const resEndDef = await envEndpoint.readPage({ operation: "identity", workspaceId: "" });
    assert.equal(resBootDef.accountId, resEndDef.accountId, "Default workspace Account ID must match between bootstrap and endpoint");
    assert.notEqual(resBootWs.accountId, resBootDef.accountId, "Different workspace must produce different account ID");

    // Whitespace-bearing user ID regression: bootstrap user.id="  whitespace-user-id  "
    // must hash identically to endpoint user.id="  whitespace-user-id  "
    const wsUserId = "  whitespace-user-id-99  ";
    const wsJwt = createSyntheticJwt({ exp: Math.floor(Date.now() / 1000) + 3600, extraClaims: { sub: wsUserId } });
    const envBootWsUser = createSyntheticEnv({
      document: createMockDocument({
        element: createBootstrapElement({
          session: { accessToken: wsJwt, user: { id: wsUserId } }
        })
      }),
      fetchHandler: async () => mockResponse({ status: 500 })
    });
    const envEndWsUser = createSyntheticEnv({
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: "tok-fallback", user: { id: wsUserId } } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const resBootWsUser = await envBootWsUser.readPage({ operation: "identity", workspaceId: "ws-1" });
    const resEndWsUser = await envEndWsUser.readPage({ operation: "identity", workspaceId: "ws-1" });
    assert.equal(resBootWsUser.accountId, resEndWsUser.accountId, "Whitespace-bearing user ID must produce identical accountId between bootstrap and endpoint");
    assert.equal(resBootWsUser.session_source, "page-bootstrap");
    assert.equal(resEndWsUser.session_source, "session-endpoint");
  }

  // Test 11: Fallbacks for missing/malformed/oversized/wrong-element/empty-field/expired/malformed-JWT
  {
    const nowSec = Math.floor(Date.now() / 1000);
    const validJwt = createSyntheticJwt({ exp: nowSec + 3600 });
    const validSession = { accessToken: validJwt, user: { id: "user-fallback-test" } };

    const cases = [
      { name: "missing element", doc: createMockDocument({ element: null }) },
      { name: "wrong tag (DIV)", doc: createMockDocument({ element: createBootstrapElement({ tagName: "DIV", session: validSession }) }) },
      { name: "wrong type (text/javascript)", doc: createMockDocument({ element: createBootstrapElement({ type: "text/javascript", session: validSession }) }) },
      { name: "oversized (>2 MiB char length)", doc: createMockDocument({ element: createBootstrapElement({ rawText: " ".repeat(2 * 1024 * 1024 + 1) }) }) },
      {
        name: "non-ASCII oversize (>2 MiB UTF-8, passes char length)",
        doc: createMockDocument({
          element: createBootstrapElement({
            rawText: JSON.stringify({
              session: validSession,
              padding: "汉".repeat(800000)
            })
          })
        })
      },
      { name: "malformed JSON", doc: createMockDocument({ element: createBootstrapElement({ rawText: '{"session": { unclosed json' }) }) },
      { name: "missing session property", doc: createMockDocument({ element: createBootstrapElement({ rawText: JSON.stringify({ notSession: 1 }) }) }) },
      { name: "empty accessToken", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: "", user: { id: "u1" } } }) }) },
      { name: "whitespace accessToken", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: "   ", user: { id: "u1" } } }) }) },
      { name: "empty user id", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: validJwt, user: { id: "" } } }) }) },
      { name: "whitespace user id", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: validJwt, user: { id: "   " } } }) }) },
      { name: "missing user object", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: validJwt } }) }) },
      { name: "expired JWT (past)", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: createSyntheticJwt({ exp: nowSec - 60 }), user: { id: "u1" } } }) }) },
      { name: "JWT expiring within 30s", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: createSyntheticJwt({ exp: nowSec + 15 }), user: { id: "u1" } } }) }) },
      { name: "JWT expiring in exactly 30s", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: createSyntheticJwt({ exp: nowSec + 30 }), user: { id: "u1" } } }) }) },
      { name: "JWT missing exp", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: createSyntheticJwt({ exp: undefined }), user: { id: "u1" } } }) }) },
      { name: "JWT non-numeric exp", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: createSyntheticJwt({ extraClaims: { exp: "tomorrow" } }), user: { id: "u1" } } }) }) },
      { name: "JWT null exp", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: createSyntheticJwt({ extraClaims: { exp: null } }), user: { id: "u1" } } }) }) },
      { name: "JWT invalid base64 in payload", doc: createMockDocument({ element: createBootstrapElement({ session: { accessToken: "header.invalid%base64.sig", user: { id: "u1" } } }) }) },
      {
        name: "JWT payload not JSON",
        doc: createMockDocument({
          element: createBootstrapElement({
            session: {
              accessToken: "header." + Buffer.from("not-json").toString("base64url") + ".sig",
              user: { id: "u1" }
            }
          })
        })
      }
    ];

    for (const tc of cases) {
      let sessionEndpointCalls = 0;
      const envFallback = createSyntheticEnv({
        document: tc.doc,
        fetchHandler: async (url) => {
          if (url.includes("/api/auth/session")) {
            sessionEndpointCalls++;
            return mockResponse({ status: 200, json: { accessToken: "endpoint-tok", user: { id: "user-endpoint" } } });
          }
          return mockResponse({ status: 200, json: {} });
        }
      });

      const res = await envFallback.readPage({ operation: "identity" });
      assert.equal(sessionEndpointCalls, 1, `Case "${tc.name}" must fall back to /api/auth/session`);
      assert.equal(res.session_source, "session-endpoint", `Case "${tc.name}" must report session-endpoint`);
      assert.match(res.accountId, /^web-[0-9a-f]{24}$/, `Case "${tc.name}" must compute valid accountId`);
    }

    // Opaque non-JWT token (not 3 segments) retains existing token validation semantics and does NOT fall back
    {
      let endpointCalls = 0;
      const opaqueDoc = createMockDocument({
        element: createBootstrapElement({
          session: { accessToken: "opaque-valid-token-string", user: { id: "user-opaque" } }
        })
      });
      const envOpaque = createSyntheticEnv({
        document: opaqueDoc,
        fetchHandler: async (url) => {
          if (url.includes("/api/auth/session")) {
            endpointCalls++;
            return mockResponse({ status: 200, json: {} });
          }
          return mockResponse({ status: 200, json: {} });
        }
      });
      const resOpaque = await envOpaque.readPage({ operation: "identity" });
      assert.equal(endpointCalls, 0, "Opaque non-JWT token must not call session endpoint");
      assert.equal(resOpaque.session_source, "page-bootstrap");
    }
  }

  // Test 12: Backend refusal not retried via another auth source
  {
    const validJwt = createSyntheticJwt({ exp: Math.floor(Date.now() / 1000) + 3600 });
    const bootstrapEl = createBootstrapElement({
      session: { accessToken: validJwt, user: { id: "user-refusal-test" } }
    });

    for (const refusalStatus of [401, 403, 429]) {
      let sessionEndpointCalls = 0;
      let historyCalls = 0;

      const envRefusal = createSyntheticEnv({
        document: createMockDocument({ element: bootstrapEl }),
        fetchHandler: async (url) => {
          if (url.includes("/api/auth/session")) {
            sessionEndpointCalls++;
            return mockResponse({ status: 200, json: { accessToken: "alternate-tok", user: { id: "user-refusal-test" } } });
          }
          if (url.includes("/backend-api/conversations")) {
            historyCalls++;
            return mockResponse({
              status: refusalStatus,
              ...(refusalStatus === 403 ? { headers: { "cf-mitigated": "challenge" } } : {})
            });
          }
          return mockResponse({ status: 200, json: {} });
        }
      });

      const res = await envRefusal.readPage({ operation: "page", limit: 20, offset: 0 });
      assert.equal(res.status, refusalStatus);
      assert.equal(res.stage, "history");
      assert.equal(res.session_source, "page-bootstrap");
      assert.equal(historyCalls, 1);
      assert.equal(sessionEndpointCalls, 0, `Backend ${refusalStatus} must not trigger alternate credential retry to session endpoint`);
      if (refusalStatus === 403) assert.equal(res.challenge, true);
    }

    // In syncHistory as well: backend 403 remains refusal without retrying session endpoint
    {
      let sessionCalls = 0;
      const envSyncRefusal = createSyntheticEnv({
        document: createMockDocument({ element: bootstrapEl }),
        storage: {
          settings: { endpoint: "http://127.0.0.1:8765", token: "tok-refuse", workspaceId: "" }
        },
        fetchHandler: async (url) => {
          if (url.includes("/api/auth/session")) {
            sessionCalls++;
            return mockResponse({ status: 200, json: {} });
          }
          if (url.includes("/backend-api/conversations")) {
            return mockResponse({ status: 403, headers: { "cf-mitigated": "challenge" } });
          }
          return mockResponse({ status: 200, json: {} });
        }
      });

      const syncResult = await envSyncRefusal.syncHistory();
      assert.equal(syncResult.ok, false);
      assert.equal(syncResult.status, 403);
      assert.equal(syncResult.stage, "history");
      assert.equal(syncResult.session_source, "page-bootstrap");
      assert.equal(sessionCalls, 0, "syncHistory backend 403 must not retry via session endpoint");

      const status = envSyncRefusal.store.get("status");
      assert.equal(status.session_source, "page-bootstrap");
      assert.match(status.detail, /Session source: page-bootstrap\./);
    }
  }

  // Test 13: Status/result secret containment and provenance
  {
    const SECRET_JWT_CLAIM = "TOP_SECRET_JWT_CLAIM_GAMMA_999";
    const SENTINEL_BOOTSTRAP_TOKEN = createSyntheticJwt({
      exp: Math.floor(Date.now() / 1000) + 3600,
      extraClaims: { private_sentinel: SECRET_JWT_CLAIM }
    });
    const SENTINEL_PAIRING_TOKEN = "SENTINEL_PAIRING_SECRET_DELTA_888";
    const SENTINEL_CONV_ID = "SENTINEL_CONV_SECRET_EPSILON_777";
    const SECRETS = [SENTINEL_BOOTSTRAP_TOKEN, SECRET_JWT_CLAIM, SENTINEL_PAIRING_TOKEN, SENTINEL_CONV_ID];

    const bootstrapEl = createBootstrapElement({
      session: { accessToken: SENTINEL_BOOTSTRAP_TOKEN, user: { id: "user-secret-test" } }
    });

    const envSecrets = createSyntheticEnv({
      document: createMockDocument({ element: bootstrapEl }),
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: SENTINEL_PAIRING_TOKEN, workspaceId: "" }
      },
      fetchHandler: async (url) => {
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({ status: 200, json: { items: [{ id: SENTINEL_CONV_ID, update_time: 1000 }] } });
        }
        if (url.includes("/backend-api/conversation/")) {
          return mockResponse({ status: 500 });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const syncRes = await envSecrets.syncHistory();
    assert.equal(syncRes.ok, false);
    assert.equal(syncRes.status, 500);
    assert.equal(syncRes.stage, "conversation");
    assert.equal(syncRes.session_source, "page-bootstrap");

    const syncResultString = JSON.stringify(syncRes);
    const storageStatusString = JSON.stringify(envSecrets.store.get("status"));

    for (const secret of SECRETS) {
      assert.equal(syncResultString.includes(secret), false, `Sync result must exclude secret: ${secret}`);
      assert.equal(storageStatusString.includes(secret), false, `Storage status must exclude secret: ${secret}`);
    }

    assert.equal(syncResultString.includes("page-bootstrap"), true);
    assert.equal(storageStatusString.includes("page-bootstrap"), true);
  }

  // Test 14: Provenance tracking across calls when source changes (mixed observed sources)
  {
    const validJwt = createSyntheticJwt({ exp: Math.floor(Date.now() / 1000) + 3600 });
    const bootstrapEl = createBootstrapElement({
      session: { accessToken: validJwt, user: { id: "user-mixed-test" } }
    });
    let lookups = 0;
    const mockDoc = {
      getElementById(id) {
        if (id === "client-bootstrap") {
          lookups++;
          return lookups === 1 ? bootstrapEl : null;
        }
        return null;
      }
    };

    let endpointCalls = 0;

    const envMixed = createSyntheticEnv({
      document: mockDoc,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-mixed", workspaceId: "" }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          endpointCalls++;
          return mockResponse({ status: 200, json: { accessToken: "tok-fallback", user: { id: "user-mixed-test" } } });
        }
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({ status: 200, json: { items: [{ id: "conv-mixed-1", update_time: 1000 }], total: 1 } });
        }
        if (url.includes("/backend-api/conversation/")) {
          return mockResponse({ status: 200, json: { mapping: { id: "root" } } });
        }
        if (url.includes("/api/import")) {
          return mockResponse({ status: 200, json: { ok: true, conversations: 1, errors: [] } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const syncRes = await envMixed.syncHistory();
    assert.equal(syncRes.ok, true);
    // Identity saw page-bootstrap, page loop saw session-endpoint
    assert.equal(syncRes.session_source, "page-bootstrap, session-endpoint");
    const status = envMixed.store.get("status");
    assert.equal(status.session_source, "page-bootstrap, session-endpoint");
    assert.match(status.detail, /Session source: page-bootstrap, session-endpoint\./);
    assert.equal(endpointCalls, 1, "Fallback must fetch session endpoint exactly once when bootstrap disappears");
  }

  // ==================== NEW EXTENDED TESTS ====================

  // Test 15: Delta and HTTP-date Retry-After parsing in readPage
  {
    const baseTime = Date.UTC(2026, 9, 27, 3, 0, 0); // 2026-10-27T03:00:00Z
    const testCases = [
      { name: "strict seconds", header: "120", expectedMs: 120000, expectedUnrep: undefined },
      { name: "seconds with whitespace", header: "  60  ", expectedMs: 60000, expectedUnrep: undefined },
      { name: "zero seconds", header: "0", expectedMs: 0, expectedUnrep: undefined },
      {
        name: "IMF-fixdate GMT",
        header: "Tue, 27 Oct 2026 03:05:00 GMT",
        expectedMs: 5 * 60 * 1000, // +5 min
        expectedUnrep: undefined
      },
      {
        name: "RFC 850 date GMT",
        header: "Tuesday, 27-Oct-26 03:10:00 GMT",
        expectedMs: 10 * 60 * 1000, // +10 min
        expectedUnrep: undefined
      },
      {
        name: "RFC 850 rolling 50-year boundary (future: 76 -> 2076)",
        header: "Tuesday, 27-Oct-76 03:00:00 GMT",
        expectedMs: Date.UTC(2076, 9, 27, 3, 0, 0) - baseTime,
        expectedUnrep: undefined
      },
      {
        name: "RFC 850 rolling 50-year boundary (51 yr future: 77 -> 1977 in past)",
        header: "Wednesday, 27-Oct-77 03:00:00 GMT",
        expectedMs: 0,
        expectedUnrep: undefined
      },
      {
        name: "RFC 850 classic 94 -> 1994 in past",
        header: "Sunday, 06-Nov-94 08:49:37 GMT",
        expectedMs: 0,
        expectedUnrep: undefined
      },
      {
        name: "ANSI C asctime treated as UTC",
        header: "Tue Oct 27 03:15:00 2026",
        expectedMs: 15 * 60 * 1000, // +15 min
        expectedUnrep: undefined
      },
      {
        name: "past HTTP-date",
        header: "Sun, 06 Nov 2020 08:49:37 GMT",
        expectedMs: 0,
        expectedUnrep: undefined
      },
      {
        name: "reject arbitrary ISO-8601 numeric string",
        header: "2026-10-27T03:05:00Z",
        expectedMs: undefined,
        expectedUnrep: undefined
      },
      {
        name: "reject garbage text",
        header: "soon",
        expectedMs: undefined,
        expectedUnrep: undefined
      },
      {
        name: "unrepresentable huge seconds",
        header: "999999999999999999999",
        expectedMs: undefined,
        expectedUnrep: true
      },
      {
        name: "valid longer delay (2 hours)",
        header: "7200",
        expectedMs: 7200000,
        expectedUnrep: undefined
      }
    ];

    for (const tc of testCases) {
      const env = createSyntheticEnv({
        initialTime: baseTime,
        fetchHandler: async () => mockResponse({
          status: 429,
          headers: { "Retry-After": tc.header }
        })
      });
      const res = await env.readPage({ operation: "identity" });
      assert.equal(res.status, 429);
      assert.equal(res.retry_after_ms, tc.expectedMs, `Case "${tc.name}" retry_after_ms mismatch`);
      assert.equal(res.retry_after_unrepresentable, tc.expectedUnrep, `Case "${tc.name}" retry_after_unrepresentable mismatch`);
    }
  }

  // Test 16: Unrepresentable long Retry-After pauses attention safely without automatic rearm
  {
    const env = createSyntheticEnv({
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1 }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-unrep" } } });
        }
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({
            status: 429,
            headers: { "Retry-After": "999999999999999999999" }
          });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const res = await env.syncHistory({ manual: false });
    assert.equal(res.ok, false);
    assert.equal(res.status, 429);

    const job = env.store.get("syncJob");
    assert.equal(job.state, "attention");
    assert.equal(job.blocked, true);
    assert.equal(job.enabled, false);
    assert.equal(job.nextAttemptAt, null);

    const settings = env.store.get("settings");
    assert.equal(settings.automatic, false);
    assert.equal(env.alarms.has("chat-recall-sync"), false, "Alarm must be cleared on unrepresentable delay");

    const status = env.store.get("status");
    assert.equal(status.title, "Import needs attention");
    assert.match(status.detail, /unrepresentable/i);
  }

  // Test 17: Valid longer server delay is honored (not capped at 30 minutes)
  {
    const baseTime = 1700000000000;
    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1 }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-longer" } } });
        }
        if (url.includes("/backend-api/conversations")) {
          // 2 hours = 7200 seconds
          return mockResponse({
            status: 429,
            headers: { "Retry-After": "7200" }
          });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const res = await env.syncHistory({ manual: false });
    assert.equal(res.ok, false);
    assert.equal(res.status, 429);

    const job = env.store.get("syncJob");
    assert.equal(job.state, "cooldown");
    assert.equal(job.attempt, 1);
    // 7200 seconds = 7,200,000 ms, must NOT be capped at 30 min (1,800,000 ms)
    assert.equal(job.nextAttemptAt, baseTime + 7200000);

    const alarm = env.alarms.get("chat-recall-sync");
    assert.ok(alarm);
    assert.equal(alarm.scheduledTime, baseTime + 7200000);
  }

  // Test 18: Exponential backoff progression and jitter bounds
  {
    const baseTime = 1700000000000;
    // Test with fixed random = 0.50 (0.50 * 0.20 = +10% jitter)
    const env = createSyntheticEnv({
      initialTime: baseTime,
      random: 0.50,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-backoff" } } });
        if (url.includes("/backend-api/conversations")) return mockResponse({ status: 429 });
        return mockResponse({ status: 200, json: {} });
      }
    });

    // Attempt 1: 60s * 1.10 = 66s = 66000ms
    await env.syncHistory({ manual: false });
    let job = env.store.get("syncJob");
    assert.equal(job.attempt, 1);
    assert.equal(job.nextAttemptAt, baseTime + 66000);

    // Advance clock past cooldown for Attempt 2
    env.setTime(job.nextAttemptAt + 1000);
    const time2 = env.currentTime();
    await env.syncHistory({ manual: false });
    job = env.store.get("syncJob");
    assert.equal(job.attempt, 2);
    // Attempt 2: 120s * 1.10 = 132s = 132000ms
    assert.equal(job.nextAttemptAt, time2 + 132000);

    // Attempt 6 (exceeding 30 min): 60s * 2^5 = 1920s > 1800s. Must be capped at 30 min (1,800,000ms)
    job.attempt = 5;
    env.store.set("syncJob", job);
    env.setTime(job.nextAttemptAt + 1000);
    const timeCap = env.currentTime();
    await env.syncHistory({ manual: false });
    job = env.store.get("syncJob");
    assert.equal(job.attempt, 6);
    assert.equal(job.nextAttemptAt, timeCap + 30 * 60 * 1000);
  }

  // Test 19: Zero remote calls before deadline (cooldown blocks manual, reset, alarm)
  {
    const baseTime = 1700000000000;
    let remoteCalls = 0;
    const cooldownDeadline = baseTime + 300000;

    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "cooldown", nextAttemptAt: cooldownDeadline, attempt: 1, revision: 1 }
      },
      fetchHandler: async () => {
        remoteCalls++;
        return mockResponse({ status: 200, json: {} });
      }
    });

    // 1. Manual syncHistory call
    const resManual = await env.syncHistory({ manual: true });
    assert.equal(resManual.ok, false);
    assert.match(resManual.error, /cooldown active/i);
    assert.equal(remoteCalls, 0, "Manual sync must make 0 remote calls during active cooldown");

    // 2. Message action: sync
    const resMsg = await env.handleMessage({ action: "sync" });
    assert.equal(resMsg.ok, false);
    assert.equal(remoteCalls, 0, "Sync message must make 0 remote calls during active cooldown");

    // 3. Reset while automatic is enabled must reject
    await assert.rejects(
      async () => env.handleMessage({ action: "reset" }),
      /disable automatic checks before resetting/i,
      "Reset action must reject while automatic enabled"
    );
    assert.equal(remoteCalls, 0, "Reset message must make 0 remote calls while automatic enabled");

    // 4. Early alarm trigger while automatic is enabled must rearm without remote calls
    await env.fireAlarm("chat-recall-sync");
    assert.equal(remoteCalls, 0, "Early alarm must make 0 remote calls and rearm deadline");
    const alarm = env.alarms.get("chat-recall-sync");
    assert.ok(alarm);
    assert.equal(alarm.scheduledTime, cooldownDeadline);

    // 5. Pause automatic through actual message; verify deadline preserved
    const pauseRes = await env.handleMessage({ action: "automatic", enabled: false });
    assert.equal(pauseRes.ok, true);
    assert.equal(pauseRes.job.state, "paused");
    assert.equal(pauseRes.job.nextAttemptAt, cooldownDeadline, "Deadline must be preserved across pause");

    // 6. With automatic paused, reset rejects due to active cooldown without remote calls
    await assert.rejects(
      async () => env.handleMessage({ action: "reset" }),
      /cooldown active/i,
      "Reset action must reject during active cooldown without remote calls"
    );
    assert.equal(remoteCalls, 0, "Reset message must make 0 remote calls during active cooldown");
  }

  // Test 20: Concurrent duplicate due alarms execute exactly once (synchronous running guard)
  {
    const baseTime = 1700000000000;
    let detailFetches = 0;
    let listFetches = 0;

    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", nextAttemptAt: baseTime, attempt: 0, revision: 1 }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-concurrent" } } });
        if (url.includes("/backend-api/conversations")) {
          listFetches++;
          return mockResponse({
            status: 200,
            json: { items: [{ id: "c-concurrent-1", update_time: 1000 }], total: 1 }
          });
        }
        if (url.includes("/backend-api/conversation/")) {
          detailFetches++;
          return mockResponse({ status: 200, json: { mapping: { id: "root" } } });
        }
        if (url.includes("/api/import")) return mockResponse({ status: 200, json: { ok: true, conversations: 1, errors: [] } });
        return mockResponse({ status: 200, json: {} });
      }
    });

    // Simulate two concurrent alarm events firing at the exact same time
    const [res1, res2] = await Promise.all([
      env.syncHistory({ manual: false }),
      env.syncHistory({ manual: false })
    ]);

    // One succeeds, one is rejected by the synchronous running guard
    const successes = [res1, res2].filter((r) => r.ok);
    const runningRejections = [res1, res2].filter((r) => !r.ok && /already running/i.test(r.error));

    assert.equal(successes.length, 1, "Exactly one invocation must succeed");
    assert.equal(runningRejections.length, 1, "Second concurrent invocation must be rejected as already running");
    assert.equal(listFetches, 1, "List must be fetched exactly once");
    assert.equal(detailFetches, 1, "Detail must be fetched exactly once without duplicate runs");
  }

  // Test 21: Automatic retry on due alarm without a click
  {
    const baseTime = 1700000000000;
    let mode = "fail429";

    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1 }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-autoretry" } } });
        if (url.includes("/backend-api/conversations")) {
          if (mode === "fail429") return mockResponse({ status: 429, headers: { "Retry-After": "60" } });
          return mockResponse({ status: 200, json: { items: [{ id: "c-retry-ok", update_time: 1000 }], total: 1 } });
        }
        if (url.includes("/backend-api/conversation/")) return mockResponse({ status: 200, json: { mapping: { id: "root" } } });
        if (url.includes("/api/import")) return mockResponse({ status: 200, json: { ok: true, conversations: 1, errors: [] } });
        return mockResponse({ status: 200, json: {} });
      }
    });

    // 1st run hits 429
    const firstRes = await env.syncHistory({ manual: false });
    assert.equal(firstRes.ok, false);
    let job = env.store.get("syncJob");
    assert.equal(job.state, "cooldown");
    const deadline = job.nextAttemptAt;

    // Advance clock past deadline and switch server mode to success
    mode = "success";
    env.setTime(deadline + 1000);

    // Alarm fires (no manual user click)
    await env.fireAlarm("chat-recall-sync");

    job = env.store.get("syncJob");
    assert.equal(job.state, "complete");
    assert.equal(job.enabled, false);

    const status = env.store.get("status");
    assert.equal(status.title, "Ordinary-list scan caught up");
  }

  // Test 22: Continued success batches and finite stop (no endless polling)
  {
    const baseTime = 1700000000000;
    let pageRequestCount = 0;

    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1 }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-batches" } } });
        if (url.includes("/backend-api/conversations")) {
          pageRequestCount++;
          const parsed = new URL(url);
          const offset = Number(parsed.searchParams.get("offset") || 0);
          if (offset === 0) {
            // Page 1: 20 items, total 25
            const items = Array.from({ length: 20 }, (_, i) => ({ id: `batch-c-${i}`, update_time: 1000 + i }));
            return mockResponse({ status: 200, json: { items, total: 25, cursor: "cur-p2" } });
          } else {
            // Page 2: 5 items, exhausted
            const items = Array.from({ length: 5 }, (_, i) => ({ id: `batch-c-${20 + i}`, update_time: 2000 + i }));
            return mockResponse({ status: 200, json: { items, total: 25, cursor: null } });
          }
        }
        if (url.includes("/backend-api/conversation/")) return mockResponse({ status: 200, json: { mapping: { id: "root" } } });
        if (url.includes("/api/import")) return mockResponse({ status: 200, json: { ok: true, conversations: 20, errors: [] } });
        return mockResponse({ status: 200, json: {} });
      }
    });

    // 1. First automatic run: reads at most ONE page (20 items)
    const run1 = await env.syncHistory({ manual: false });
    assert.equal(run1.ok, true);
    assert.equal(run1.imported, 20);
    assert.equal(pageRequestCount, 1);

    let job = env.store.get("syncJob");
    assert.equal(job.state, "waiting");
    assert.ok(job.nextAttemptAt >= baseTime + 60000, "Next batch must be scheduled no sooner than 60s");

    const checkpoint = env.store.get(`checkpoint:${job.accountId}`);
    assert.equal(checkpoint.offset, 20);

    // 2. Advance clock by 60s and fire alarm for next batch
    env.setTime(job.nextAttemptAt);
    await env.fireAlarm("chat-recall-sync");
    assert.equal(pageRequestCount, 2);

    job = env.store.get("syncJob");
    assert.equal(job.state, "complete");
    assert.equal(job.enabled, false);

    const cfg = env.store.get("settings");
    assert.equal(cfg.automatic, false, "Automatic mode must turn off on complete");
    assert.equal(env.alarms.has("chat-recall-sync"), false, "All alarms cleared after finite stop");
  }

  // Test 23: Service worker restart / crash recovery from "running" state
  {
    const baseTime = 1700000000000;
    const persistentStore = new Map();
    persistentStore.set("settings", { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true });
    // Died while in "running" state:
    persistentStore.set("syncJob", { version: 1, enabled: true, state: "running", attempt: 0, revision: 1 });

    const newWorkerEnv = createSyntheticEnv({
      initialTime: baseTime,
      store: persistentStore,
      fetchHandler: async () => mockResponse({ status: 200, json: {} })
    });

    // Worker starts up / receives status query
    const statusRes = await newWorkerEnv.handleMessage({ action: "status" });
    assert.equal(statusRes.job.state, "waiting", "Dead running state must recover to waiting");
    assert.equal(statusRes.job.enabled, true);

    const alarm = newWorkerEnv.alarms.get("chat-recall-sync");
    assert.ok(alarm, "Alarm must be reconstructed for recovered job");
  }

  // Test 24: Pause during in-flight run preserves cooldown and cannot be undone
  {
    const baseTime = 1700000000000;
    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "cooldown", nextAttemptAt: baseTime + 120000, attempt: 2, revision: 1 }
      },
      fetchHandler: async () => mockResponse({ status: 200, json: {} })
    });

    // User disables automatic mode
    const pauseRes = await env.handleMessage({ action: "automatic", enabled: false });
    assert.equal(pauseRes.ok, true);
    assert.equal(pauseRes.job.state, "paused");
    assert.equal(pauseRes.job.nextAttemptAt, baseTime + 120000, "Cooldown deadline must be preserved across pause");
    assert.equal(pauseRes.job.attempt, 2, "Attempt must be preserved across pause");

    const cfg = env.store.get("settings");
    assert.equal(cfg.automatic, false);
    assert.equal(env.alarms.has("chat-recall-sync"), false, "Alarm must be cleared on pause");
  }

  // Test 25: Changed config stops auto, bumps revision, and preserves cooldown
  {
    const baseTime = 1700000000000;
    const cooldownDeadline = baseTime + 180000;

    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-old", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "cooldown", nextAttemptAt: cooldownDeadline, attempt: 1, revision: 1 }
      },
      fetchHandler: async () => mockResponse({ status: 200, json: {} })
    });

    const saveRes = await env.handleMessage({
      action: "save",
      settings: { endpoint: "http://127.0.0.1:8765", token: "tok-new", workspaceId: "ws-new" }
    });
    assert.equal(saveRes.ok, true);

    const cfg = env.store.get("settings");
    assert.equal(cfg.automatic, false, "Saving pairing must stop automatic");
    assert.equal(cfg.token, "tok-new");
    assert.equal(cfg.workspaceId, "ws-new");

    const job = env.store.get("syncJob");
    assert.equal(job.revision, 2, "Revision must increment on config change");
    assert.equal(job.state, "paused");
    assert.equal(job.nextAttemptAt, cooldownDeadline, "Cooldown deadline must carry conservatively");
  }

  // Test 26: Account change during auto run pauses in attention before history import
  {
    let historyCalls = 0;
    let detailCalls = 0;
    let localImportCalls = 0;

    const env = createSyntheticEnv({
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1, accountId: "web-original-account" }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "user-different" } } });
        }
        if (url.includes("/backend-api/conversations")) {
          historyCalls++;
          return mockResponse({ status: 200, json: { items: [] } });
        }
        if (url.includes("/backend-api/conversation/")) {
          detailCalls++;
          return mockResponse({ status: 200, json: {} });
        }
        if (url.includes("/api/import")) {
          localImportCalls++;
          return mockResponse({ status: 200, json: {} });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const res = await env.syncHistory({ manual: false });
    assert.equal(res.ok, false);
    assert.match(res.error, /account changed/i);

    // Assert zero history, detail, and local import requests were made
    assert.equal(historyCalls, 0, "No history requests must be made after account change");
    assert.equal(detailCalls, 0, "No detail requests must be made after account change");
    assert.equal(localImportCalls, 0, "No local import requests must be made after account change");

    const job = env.store.get("syncJob");
    assert.equal(job.state, "attention", "Account change must transition to attention");
    assert.equal(job.enabled, false);
    assert.equal(job.accountId, "web-original-account", "Original accountId must remain pinned");

    const cfg = env.store.get("settings");
    assert.equal(cfg.automatic, false, "Automatic mode must be disabled on account change");
    assert.equal(env.alarms.has("chat-recall-sync"), false, "All alarms must be cleared on account change");
  }

  // Test 27: Non-429 errors (401, 403, challenge, local import error) pause in attention without retry
  {
    const non429Cases = [
      { name: "401 auth error", status: 401, headers: {} },
      { name: "403 challenge error", status: 403, headers: { "cf-mitigated": "challenge" } },
      { name: "500 server error", status: 500, headers: {} }
    ];

    for (const tc of non429Cases) {
      const env = createSyntheticEnv({
        storage: {
          settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
          syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1 }
        },
        fetchHandler: async (url) => {
          if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-non429" } } });
          if (url.includes("/backend-api/conversations")) return mockResponse({ status: tc.status, headers: tc.headers });
          return mockResponse({ status: 200, json: {} });
        }
      });

      const res = await env.syncHistory({ manual: false });
      assert.equal(res.ok, false);

      const job = env.store.get("syncJob");
      assert.equal(job.state, "attention", `Case "${tc.name}" must transition to attention`);
      assert.equal(job.enabled, false);
      assert.equal(env.alarms.has("chat-recall-sync"), false, `Case "${tc.name}" must not schedule retry alarm`);
    }
  }

  // Test 28: Partial-page detail 429 commits previous conversations, merges versions, retries skipping saved
  {
    const baseTime = 1700000000000;
    let detailMode = "c3-fail";
    const fetchedDetails = [];
    const localImportedBatches = [];

    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1 }
      },
      fetchHandler: async (url, opts) => {
        if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-partial" } } });
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({
            status: 200,
            json: {
              items: [
                { id: "c1", update_time: 1000 },
                { id: "c2", update_time: 2000 },
                { id: "c3", update_time: 3000 }
              ],
              total: 3
            }
          });
        }
        if (url.includes("/backend-api/conversation/")) {
          const convId = url.split("/").pop();
          fetchedDetails.push(convId);
          if (convId === "c3" && detailMode === "c3-fail") {
            return mockResponse({ status: 429, headers: { "Retry-After": "60" } });
          }
          return mockResponse({ status: 200, json: { mapping: { id: `m-${convId}` } } });
        }
        if (url.includes("/api/import")) {
          const body = JSON.parse(opts?.body || "{}");
          localImportedBatches.push(body.payload || []);
          return mockResponse({ status: 200, json: { ok: true, conversations: (body.payload || []).length, errors: [] } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    // First attempt: c1 and c2 succeed, c3 returns 429
    const res1 = await env.syncHistory({ manual: false });
    assert.equal(res1.ok, false);
    assert.equal(res1.status, 429);
    assert.equal(res1.stage, "conversation");

    // Checkpoint verification: offset & pages MUST NOT advance
    const job1 = env.store.get("syncJob");
    assert.equal(job1.state, "cooldown");
    assert.equal(job1.attempt, 1);

    const checkpoint1 = env.store.get(`checkpoint:${job1.accountId}`);
    assert.equal(checkpoint1.offset, 0, "Checkpoint offset must not advance on failed page");
    assert.equal(checkpoint1.pages, 0, "Checkpoint pages must not advance on failed page");
    // But versions for c1 and c2 were merged!
    assert.equal(checkpoint1.versions["c1"], "1000000");
    assert.equal(checkpoint1.versions["c2"], "2000000");
    assert.equal(checkpoint1.versions["c3"], undefined);

    // Verify local import was called with [c1, c2]
    assert.equal(localImportedBatches.length, 1);
    assert.equal(localImportedBatches[0].length, 2);
    assert.deepEqual(localImportedBatches[0].map((c) => c.id), ["c1", "c2"]);

    // Advance clock past cooldown and retry via alarm
    detailMode = "c3-ok";
    fetchedDetails.length = 0; // reset tracker
    localImportedBatches.length = 0;
    env.setTime(job1.nextAttemptAt + 1000);

    await env.fireAlarm("chat-recall-sync");

    // c1 and c2 are in checkpoint.versions, so their detail fetch MUST BE SKIPPED!
    assert.deepEqual(fetchedDetails, ["c3"], "Only c3 detail must be fetched on retry; c1 and c2 must be skipped");

    // Local import called with only c3
    assert.equal(localImportedBatches.length, 1);
    assert.deepEqual(localImportedBatches[0].map((c) => c.id), ["c3"]);

    // Complete scan caught up!
    const finalJob = env.store.get("syncJob");
    assert.equal(finalJob.state, "complete");
    assert.equal(finalJob.attempt, 0);

    const finalCheckpoint = env.store.get(`checkpoint:${job1.accountId}`);
    assert.equal(finalCheckpoint.offset, 0); // 0 because completed resets offset to 0 for poll phase
    assert.equal(finalCheckpoint.phase, "poll");
    assert.equal(finalCheckpoint.versions["c3"], "3000000");
  }

  // Test 29: Partial local failure retains/persists 429 backoff deadline, leaves checkpoint unchanged, and pauses attention without alarms
  {
    const baseTime = 1700000000000;
    let remoteCalls = 0;
    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1 }
      },
      fetchHandler: async (url) => {
        remoteCalls++;
        if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-partfail" } } });
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({
            status: 200,
            json: { items: [{ id: "c1", update_time: 1000 }, { id: "c2", update_time: 2000 }], total: 2 }
          });
        }
        if (url.includes("/backend-api/conversation/")) {
          const convId = url.split("/").pop();
          if (convId === "c2") return mockResponse({ status: 429, headers: { "Retry-After": "3600" } });
          return mockResponse({ status: 200, json: { mapping: { id: "root" } } });
        }
        if (url.includes("/api/import")) {
          // Local import fails!
          return mockResponse({ status: 500, json: { error: "Database disk full" } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const res = await env.syncHistory({ manual: false });
    assert.equal(res.ok, false);
    assert.equal(res.status, 429);

    const job = env.store.get("syncJob");
    assert.equal(job.state, "attention", "Local import failure must pause in attention");
    assert.equal(job.enabled, false);
    assert.equal(job.attempt, 1, "Attempt must be incremented on 429 even when local import fails");
    assert.equal(job.nextAttemptAt, baseTime + 3600000, "Retry-After 3600 deadline must be persisted");
    assert.equal(env.alarms.has("chat-recall-sync"), false, "No alarm should be scheduled when paused in attention");

    const checkpoint = env.store.get(`checkpoint:${job.accountId}`);
    assert.equal(checkpoint, undefined, "No checkpoint should be saved when local import fails");

    // Zero-network guards: manual sync and reset must be blocked by the persisted cooldown
    remoteCalls = 0;
    const resManual = await env.syncHistory({ manual: true });
    assert.equal(resManual.ok, false);
    assert.match(resManual.error, /cooldown active/i);
    assert.equal(remoteCalls, 0, "Manual sync must make 0 remote calls during active cooldown after partial import failure");

    await assert.rejects(
      async () => env.handleMessage({ action: "reset" }),
      /cooldown active/i,
      "Reset action must reject during active cooldown without remote calls"
    );
    assert.equal(remoteCalls, 0, "Reset must make 0 remote calls during active cooldown after partial import failure");
  }

  // Test 30: Zero credential leakage in job, status, or results
  {
    const SECRET_ACCESS_TOKEN = "SECRET_BEARER_TOKEN_999";
    const SECRET_PAIRING_TOKEN = "SECRET_PAIRING_TOKEN_888";
    const env = createSyntheticEnv({
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: SECRET_PAIRING_TOKEN, workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1 }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          return mockResponse({ status: 200, json: { accessToken: SECRET_ACCESS_TOKEN, user: { id: "u-leak" } } });
        }
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({ status: 429, headers: { "Retry-After": "60" } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    const res = await env.syncHistory({ manual: false });
    const storedJob = env.store.get("syncJob");
    const storedStatus = env.store.get("status");

    const jsonRes = JSON.stringify(res);
    const jsonJob = JSON.stringify(storedJob);
    const jsonStatus = JSON.stringify(storedStatus);

    for (const secret of [SECRET_ACCESS_TOKEN, SECRET_PAIRING_TOKEN]) {
      assert.equal(jsonRes.includes(secret), false, `Result must not contain ${secret}`);
      assert.equal(jsonJob.includes(secret), false, `Job must not contain ${secret}`);
      assert.equal(jsonStatus.includes(secret), false, `Status must not contain ${secret}`);
    }

    assert.equal(jsonJob.includes("authorization"), false);
    assert.equal(jsonJob.includes("headers"), false);
  }

  // Test 31: Deferred detail fetch with automatic paused in flight resolves as HTTP 429 Retry-After 3600
  // Merges server cooldown into fresh job, leaves job paused with NO alarm, and blocks manual sync until deadline
  {
    const baseTime = 1700000000000;
    let resolveDetail;
    const detailPromise = new Promise((resolve) => {
      resolveDetail = resolve;
    });
    let notifyDetailStarted;
    const detailStarted = new Promise((resolve) => {
      notifyDetailStarted = resolve;
    });

    let remoteCalls = 0;
    let detailCalls = 0;

    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-test", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1 }
      },
      fetchHandler: async (url) => {
        remoteCalls++;
        if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-deferred" } } });
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({ status: 200, json: { items: [{ id: "c-def-1", update_time: 1000 }], total: 1 } });
        }
        if (url.includes("/backend-api/conversation/")) {
          detailCalls++;
          if (detailCalls === 1) {
            notifyDetailStarted();
            return detailPromise;
          }
          return mockResponse({ status: 200, json: { mapping: { id: "root" } } });
        }
        if (url.includes("/api/import")) {
          return mockResponse({ status: 200, json: { ok: true, conversations: 1, errors: [] } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    // Start auto syncHistory in background
    const syncPromise = env.syncHistory({ manual: false });

    // Explicitly await that detail fetch has entered before pausing
    await detailStarted;

    // While in flight, user pauses automatic mode via message
    const pauseRes = await env.handleMessage({ action: "automatic", enabled: false });
    assert.equal(pauseRes.ok, true);
    assert.equal(pauseRes.job.enabled, false);
    assert.equal(pauseRes.job.state, "paused");
    const pausedRevision = pauseRes.job.revision;
    assert.ok(pausedRevision > 1, "Revision must increment on pause");

    // Now resolve detail fetch with 429 Retry-After 3600
    resolveDetail(mockResponse({
      status: 429,
      headers: { "Retry-After": "3600" }
    }));

    const syncRes = await syncPromise;
    assert.equal(syncRes.ok, false);
    assert.equal(syncRes.status, 429);

    // Job in storage MUST be merged into fresh job:
    const job = env.store.get("syncJob");
    assert.equal(job.enabled, false, "Job must remain paused/disabled");
    assert.equal(job.state, "paused", "Job state must remain paused");
    assert.equal(job.revision, pausedRevision, "Job revision must not be reverted");
    assert.equal(job.attempt, 1, "Attempt must be recorded from 429");
    assert.equal(job.nextAttemptAt, baseTime + 3600000, "Deadline must be merged from 429");

    // NO alarm should be scheduled because job is paused
    assert.equal(env.alarms.has("chat-recall-sync"), false, "Paused job must have NO alarm");

    // Reset remote calls counter to verify zero calls before deadline
    remoteCalls = 0;
    assert.equal(detailCalls, 1, "Exactly one detail fetch occurred during initial in-flight run");

    // CRITICAL: Manual retry must be BLOCKED before deadline
    const resManualBlocked = await env.syncHistory({ manual: true });
    assert.equal(resManualBlocked.ok, false);
    assert.match(resManualBlocked.error, /cooldown active/i);
    assert.equal(remoteCalls, 0, "No remote calls permitted during cooldown for manual sync");
    assert.equal(detailCalls, 1, "No earlier detail fetch permitted during cooldown for manual sync");

    // Message action sync must also be blocked
    const resMsgBlocked = await env.handleMessage({ action: "sync" });
    assert.equal(resMsgBlocked.ok, false);
    assert.match(resMsgBlocked.error, /cooldown active/i);
    assert.equal(remoteCalls, 0, "No remote calls permitted during cooldown for sync message");
    assert.equal(detailCalls, 1, "No earlier detail fetch permitted during cooldown for sync message");

    // Reset action must be rejected
    await assert.rejects(
      async () => env.handleMessage({ action: "reset" }),
      /cooldown active/i,
      "Reset action must reject during active cooldown without remote calls"
    );
    assert.equal(remoteCalls, 0, "No remote calls permitted during cooldown for reset message");
    assert.equal(detailCalls, 1, "No earlier detail fetch permitted during cooldown for reset message");

    // Advance past deadline -> manual retry is now permitted
    env.setTime(baseTime + 3600000 + 1000);
    // When deadline passes, manual sync can proceed
    const resAfterDeadline = await env.syncHistory({ manual: true });
    assert.equal(resAfterDeadline.ok, true);
    assert.ok(remoteCalls > 0, "Remote calls permitted once deadline has passed");
    assert.equal(detailCalls, 2, "Exactly one additional detail fetch permitted after deadline");
  }

  // Test 32: Deferred detail fetch with config changed (save pairing) in flight resolves as 429
  // Merges server cooldown into fresh job, does NOT import to new config, leaves job paused with NO alarm, and blocks manual sync until deadline
  {
    const baseTime = 1700000000000;
    let resolveDetail;
    const detailPromise = new Promise((resolve) => {
      resolveDetail = resolve;
    });
    let notifyDetailStarted;
    const detailStarted = new Promise((resolve) => {
      notifyDetailStarted = resolve;
    });

    let localImportCalls = 0;
    let remoteCalls = 0;

    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-original", workspaceId: "", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1 }
      },
      fetchHandler: async (url) => {
        remoteCalls++;
        if (url.includes("/api/auth/session")) return mockResponse({ status: 200, json: { accessToken: "tok", user: { id: "u-cfgchange" } } });
        if (url.includes("/backend-api/conversations")) {
          return mockResponse({ status: 200, json: { items: [{ id: "c-cfg-1", update_time: 1000 }], total: 1 } });
        }
        if (url.includes("/backend-api/conversation/")) {
          notifyDetailStarted();
          return detailPromise;
        }
        if (url.includes("/api/import")) {
          localImportCalls++;
          return mockResponse({ status: 200, json: { ok: true, conversations: 1, errors: [] } });
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    // Start auto syncHistory in background
    const syncPromise = env.syncHistory({ manual: false });

    // Explicitly await that detail fetch has entered before saving new config
    await detailStarted;

    // While in flight, user saves new pairing settings
    const saveRes = await env.handleMessage({
      action: "save",
      settings: { endpoint: "http://127.0.0.1:8765", token: "tok-new-pair", workspaceId: "ws-new" }
    });
    assert.equal(saveRes.ok, true);
    assert.equal(saveRes.job.enabled, false);
    assert.equal(saveRes.job.state, "paused");
    const savedRevision = saveRes.job.revision;

    // Resolve detail with 429 Retry-After 1800
    resolveDetail(mockResponse({
      status: 429,
      headers: { "Retry-After": "1800" }
    }));

    const syncRes = await syncPromise;
    assert.equal(syncRes.ok, false);
    assert.equal(syncRes.status, 429);

    // Verify stale context was NOT imported to new endpoint/token
    assert.equal(localImportCalls, 0, "Stale context payload must not be imported after config change");

    // Job in storage has merged deadline, preserved revision and paused state
    const job = env.store.get("syncJob");
    assert.equal(job.enabled, false);
    assert.equal(job.state, "paused");
    assert.equal(job.revision, savedRevision);
    assert.equal(job.attempt, 1);
    assert.equal(job.nextAttemptAt, baseTime + 1800000);
    assert.equal(env.alarms.has("chat-recall-sync"), false);

    // Immediate manual retry is blocked
    remoteCalls = 0;
    const resManual = await env.syncHistory({ manual: true });
    assert.equal(resManual.ok, false);
    assert.match(resManual.error, /cooldown active/i);
    assert.equal(remoteCalls, 0, "No remote calls permitted during cooldown after config change");
  }

  // Test 33: Deferred identity with pause and config changes in flight does not overwrite job/settings with stale writes
  {
    const baseTime = 1700000000000;
    let resolveIdentity;
    const identityPromise = new Promise((resolve) => {
      resolveIdentity = resolve;
    });
    let notifyIdentityStarted;
    const identityStarted = new Promise((resolve) => {
      notifyIdentityStarted = resolve;
    });

    const env = createSyntheticEnv({
      initialTime: baseTime,
      storage: {
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok-initial", workspaceId: "ws-orig", automatic: true },
        syncJob: { version: 1, enabled: true, state: "waiting", attempt: 0, revision: 1, accountId: null }
      },
      fetchHandler: async (url) => {
        if (url.includes("/api/auth/session")) {
          notifyIdentityStarted();
          return identityPromise;
        }
        return mockResponse({ status: 200, json: {} });
      }
    });

    // Start auto syncHistory (suspends awaiting identity check)
    const syncPromise = env.syncHistory({ manual: false });

    // Explicitly await that identity fetch has entered before saving new config
    await identityStarted;

    // While identity is in flight, user pauses and changes config
    const saveRes = await env.handleMessage({
      action: "save",
      settings: { endpoint: "http://127.0.0.1:8765", token: "tok-updated", workspaceId: "ws-updated" }
    });
    assert.equal(saveRes.ok, true);
    assert.equal(saveRes.job.enabled, false);
    assert.equal(saveRes.job.state, "paused");
    const savedRevision = saveRes.job.revision;

    // Resolve deferred identity
    resolveIdentity(mockResponse({
      status: 200,
      json: { accessToken: "tok-page-auth", user: { id: "user-deferred-id" } }
    }));

    const syncRes = await syncPromise;
    assert.equal(syncRes.ok, false);

    // Assert that settings in storage were NOT overwritten by stale identity finish
    const finalCfg = env.store.get("settings");
    assert.equal(finalCfg.token, "tok-updated", "Settings token must not be overwritten by stale identity write");
    assert.equal(finalCfg.workspaceId, "ws-updated", "Settings workspaceId must not be overwritten by stale identity write");
    assert.equal(finalCfg.automatic, false);

    // Assert that job in storage was NOT overwritten (remains paused, revision preserved, accountId not pinned to stale run)
    const finalJob = env.store.get("syncJob");
    assert.equal(finalJob.revision, savedRevision, "Job revision must be preserved");
    assert.equal(finalJob.state, "paused", "Job state must remain paused");
    assert.equal(finalJob.enabled, false, "Job must remain disabled");
    assert.equal(finalJob.accountId, null, "accountId must not be pinned on stale job");
  }

  console.log("Synthetic VM offline checks passed (session 401/403/challenge/429, malformed JSON, history/detail failures, no token leakage, zero checkpoint advancement on failure, explicit retry succeeding, page-bootstrap zero-fetch/fallbacks/provenance, delta/HTTP-date Retry-After, exponential backoff/jitter bounds, cooldown blocking, concurrent duplicate due alarms synchronous guard, automatic retry without click, continued batches & finite stop, SW restart recovery, pause cooldown preservation, config/account change safety, partial 429 commit & retry skip, partial local failure, zero credential leakage, deferred fetch pause/save 429 race preservation, deferred identity race preservation).");
}

await runSyntheticTests();
