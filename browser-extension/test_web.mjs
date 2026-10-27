// Offline DOM-safety / cross-language offset tests, without a browser dependency.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

class Node {
  constructor(tag, text = "") { this.tag = tag; this.children = []; this.textContent = text; }
  append(...children) { this.children.push(...children); }
  get text() { return this.textContent + this.children.map((child) => child.text).join(""); }
}
const document = { createElement: (tag) => new Node(tag), createTextNode: (text) => new Node("#text", text) };
const script = await readFile(new URL("../chat_recall/web/app.js", import.meta.url), "utf8");
const functions = script.slice(script.indexOf("function element("), script.indexOf("function notice(")) + script.slice(script.indexOf("function appendHighlighted("), script.indexOf("function emptyState(")) + script.slice(script.indexOf("function semanticText("), script.indexOf("async function refreshStatus("));
const context = vm.createContext({ document });
vm.runInContext(`${functions};globalThis.highlight = appendHighlighted;globalThis.readiness = semanticText;globalThis.semanticJobText = semanticJobText;`, context);
const target = new Node("p");
context.highlight(target, "A😀straße😀B", [[2, 8]], false);
assert.equal(target.text, "A😀straße😀B");
assert.equal(target.children.find((node) => node.tag === "mark").text, "straße");
const overlap = new Node("p");
context.highlight(overlap, "banana", [[1, 4], [3, 6]], false);
assert.equal(overlap.text, "banana");
assert.equal(overlap.children.filter((node) => node.tag === "mark").length, 1);
assert.equal(overlap.children.find((node) => node.tag === "mark").text, "anana");
const unsafe = new Node("p");
context.highlight(unsafe, '<img src=x onerror="alert(1)">', [[1, 4]], false);
assert.equal(unsafe.text, '<img src=x onerror="alert(1)">');
assert(unsafe.children.every((node) => ["#text", "mark"].includes(node.tag)));
const excerpt = new Node("p");
context.highlight(excerpt, "😀".repeat(400), [[350, 351]], true);
assert.equal(excerpt.children.find((node) => node.tag === "mark").text, "😀");
assert(!excerpt.text.includes(String.fromCodePoint(0xFFFD)));

const longPrefix = "BEGINNING_ONLY " + "x".repeat(1800);
const semanticPassage = "The relevant dialogue is near this message's end.";
const longText = longPrefix + semanticPassage;
const semanticExcerpt = new Node("p");
context.highlight(semanticExcerpt, longText, [], true, Array.from(longPrefix).length);
assert(semanticExcerpt.text.includes(semanticPassage));
assert(!semanticExcerpt.text.includes("BEGINNING_ONLY"));
assert.equal(semanticExcerpt.children.filter((node) => node.tag === "mark").length, 0);

const astralText = "😀".repeat(1000) + "SEMANTIC_FOCUS" + "尾".repeat(1000);
const astralExcerpt = new Node("p");
context.highlight(astralExcerpt, astralText, [], true, 1000);
assert(astralExcerpt.text.includes("SEMANTIC_FOCUS"));
assert(!astralExcerpt.text.includes(String.fromCodePoint(0xFFFD)));

const literalFirst = new Node("p");
context.highlight(literalFirst, longText, [[0, 14]], true, Array.from(longPrefix).length);
assert(literalFirst.text.includes("BEGINNING_ONLY"));
assert(!literalFirst.text.includes(semanticPassage));
assert.equal(literalFirst.children.find((node) => node.tag === "mark").text, "BEGINNING_ONLY");

for (const invalidFocus of [undefined, null, "1800", NaN, Infinity, 12.5, -50]) {
  const fallback = new Node("p");
  context.highlight(fallback, longText, [], true, invalidFocus);
  assert(fallback.text.startsWith("BEGINNING_ONLY"));
}
const clamped = new Node("p");
context.highlight(clamped, longText, [], true, 9999999);
assert(clamped.text.includes(semanticPassage));
const fullReader = new Node("p");
context.highlight(fullReader, astralText, [], false, 1000);
assert.equal(fullReader.text, astralText);

assert(!/\binnerHTML\s*=|\bouterHTML\s*=|document\.write\s*\(/.test(script));
assert.equal(context.readiness({ configured: true, model_id: "selected", models: [{ model_id: "selected", indexed_messages: 2, vectors: 4 }] }), "2 messages ready for meaning search");
assert.equal(context.readiness({ configured: true, model_id: "selected", models: [{ model_id: "unrelated", indexed_messages: 50, vectors: 50 }] }), "Meaning search not ready · index the configured model");
assert.equal(context.readiness({ configured: false, model_id: null, models: [{ model_id: "unrelated", indexed_messages: 50, vectors: 50 }] }), "Meaning search not configured · keyword search available");

// Offline job diagnostics, running state, partial/failed inert text and success behavior.
assert.equal(context.semanticJobText({ state: "running" }), "Background meaning indexing in progress");
const readySemantic = { configured: true, model_id: "selected", models: [{ model_id: "selected", indexed_messages: 2, vectors: 4 }] };
assert.equal(context.semanticJobText({ state: "complete", report: { processed: 2, remaining: 0, errors: [] } }, readySemantic), "2 messages ready for meaning search");
assert.equal(context.semanticJobText({ state: "idle" }, readySemantic), "2 messages ready for meaning search");
assert.equal(context.semanticJobText(null, readySemantic), "2 messages ready for meaning search");

const partialJob = {
  state: "partial",
  report: {
    processed: 5,
    remaining: 3,
    errors: [{ conversation_id: "c1", message_id: "m1", error: "embedding provider rate limit" }],
  },
};
assert.equal(context.semanticJobText(partialJob), "Partial meaning indexing: embedding provider rate limit");

const failedJobDict = {
  state: "failed",
  report: {
    processed: 0,
    remaining: 1,
    errors: [{ conversation_id: "c1", message_id: "m1", error: "embedding provider offline" }],
  },
};
assert.equal(context.semanticJobText(failedJobDict), "Meaning indexing needs attention: embedding provider offline");

const failedJobString = {
  state: "failed",
  report: {
    processed: 0,
    remaining: 1,
    errors: ["plain string error message"],
  },
};
assert.equal(context.semanticJobText(failedJobString), "Meaning indexing needs attention: plain string error message");

const failedJobException = {
  state: "failed",
  error: "synthetic database unavailable",
};
assert.equal(context.semanticJobText(failedJobException), "Meaning indexing needs attention: synthetic database unavailable");

assert.equal(context.semanticJobText({ state: "failed" }), "Meaning indexing needs attention: unknown error");
assert.equal(context.semanticJobText({ state: "partial", report: { errors: [] } }), "Partial meaning indexing: unknown error");

const longError = "E".repeat(300);
const boundedJob = {
  state: "failed",
  report: { errors: [{ error: longError }] },
};
assert.equal(context.semanticJobText(boundedJob), `Meaning indexing needs attention: ${"E".repeat(240)}`);

const maliciousJob = {
  state: "failed",
  report: { errors: [{ error: '<img src=x onerror="alert(1)"> & <script>' }] },
};
const statusNode = new Node("span");
statusNode.textContent = context.semanticJobText(maliciousJob);
assert.equal(statusNode.children.length, 0);
assert.equal(statusNode.text, 'Meaning indexing needs attention: <img src=x onerror="alert(1)"> & <script>');

console.log("Web rendering offline checks passed (Unicode offsets, literal/semantic focus, inert markup, selected-model readiness, job diagnostics).");

// --- Synthetic VM execution tests for popup.js ---
class PopupSyntheticElement {
  constructor(id) {
    this.id = id;
    this.textContent = "";
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.files = [];
    this._classes = new Set();
    this._listeners = {};
    const self = this;
    this.classList = {
      add(c) { self._classes.add(c); },
      remove(c) { self._classes.delete(c); },
      toggle(c, force) {
        if (force === undefined) {
          if (self._classes.has(c)) self._classes.delete(c);
          else self._classes.add(c);
        } else if (force) {
          self._classes.add(c);
        } else {
          self._classes.delete(c);
        }
      },
      contains(c) { return self._classes.has(c); },
    };
  }
  addEventListener(event, listener) {
    if (!this._listeners[event]) this._listeners[event] = [];
    this._listeners[event].push(listener);
  }
  async dispatchEvent(event) {
    const type = typeof event === "string" ? event : event.type;
    const ev = typeof event === "string" ? { type: event, preventDefault() {} } : event;
    if (!ev.preventDefault) ev.preventDefault = () => {};
    const list = this._listeners[type] || [];
    for (const fn of list) {
      await fn(ev);
    }
  }
}

const popupScript = await readFile(new URL("./popup.js", import.meta.url), "utf8");
assert(!/\binnerHTML\s*=|\bouterHTML\s*=|document\.write\s*\(/.test(popupScript), "popup.js must not use innerHTML/outerHTML");

async function setupPopup(messageHandler) {
  const elements = {};
  const getEl = (id) => {
    if (!elements[id]) elements[id] = new PopupSyntheticElement(id);
    return elements[id];
  };
  const intervals = [];
  const fakeSetInterval = (fn, ms) => {
    intervals.push({ fn, ms });
    return intervals.length;
  };
  const fakeClearInterval = () => {};
  const windowObj = {
    addEventListener: () => {},
  };
  const docObj = {
    getElementById: getEl,
  };
  const chromeObj = {
    runtime: {
      id: "synthetic-popup-ext-id",
      sendMessage: async (msg) => {
        const payload = msg !== undefined && msg !== null ? JSON.parse(JSON.stringify(msg)) : msg;
        const res = await messageHandler(payload);
        return res !== undefined && res !== null ? JSON.parse(JSON.stringify(res)) : res;
      },
    },
  };

  const popupContext = vm.createContext({
    document: docObj,
    window: windowObj,
    chrome: chromeObj,
    setInterval: fakeSetInterval,
    clearInterval: fakeClearInterval,
    Date,
    String,
    Number,
    Boolean,
    Error,
    JSON,
  });

  vm.runInContext(popupScript, popupContext);
  await new Promise((resolve) => setImmediate(resolve));

  return { elements, getEl, intervals };
}

// 1. Queued 429 preserves background diagnostic/source and retry deadline; manual button disabled while deadline future.
{
  const futureDeadline = Date.now() + 120000;
  let bgStatus = {
    title: "Ready",
    detail: "Connected to local service (source: paired-host)",
    error: false,
    lastSuccess: Date.now() - 60000,
  };
  let bgJob = {
    enabled: true,
    state: "waiting",
    nextAttemptAt: null,
    attempt: 0,
  };
  let bgSettings = {
    endpoint: "http://127.0.0.1:8765",
    token: "token-abc",
    workspaceId: "ws-1",
    automatic: true,
  };

  const handler = async (msg) => {
    if (msg.action === "status") {
      return {
        status: bgStatus,
        settings: bgSettings,
        running: false,
        job: bgJob,
      };
    }
    if (msg.action === "sync") {
      // Structured 429 response from background: background stores authoritative backoff and safe provenance
      bgStatus = {
        title: "ChatGPT rate limit 429",
        detail: "HTTP 429: Too Many Requests · retry after cooldown (source: chatgpt-api-sync)",
        error: true,
        lastSuccess: Date.now() - 60000,
      };
      bgJob = {
        enabled: true,
        state: "cooldown",
        nextAttemptAt: futureDeadline,
        attempt: 2,
      };
      return { ok: false, error: "429 Too Many Requests" };
    }
    return { ok: true };
  };

  const { getEl } = await setupPopup(handler);

  // Initial load populated correctly
  assert.equal(getEl("extension-origin").textContent, "chrome-extension://synthetic-popup-ext-id");
  assert.equal(getEl("endpoint").value, "http://127.0.0.1:8765");
  assert.equal(getEl("token").value, "token-abc");
  assert.equal(getEl("workspace-id").value, "ws-1");
  assert.equal(getEl("automatic").checked, true);
  assert.equal(getEl("sync").disabled, false);

  // User initiates sync, which encounters structured 429
  await getEl("sync").dispatchEvent("click");

  // Verify background diagnostic/source is preserved rather than overwritten by "Needs attention" / response.error
  assert.equal(getEl("status-title").textContent, "ChatGPT rate limit 429");
  assert.equal(getEl("status-detail").textContent, "HTTP 429: Too Many Requests · retry after cooldown (source: chatgpt-api-sync)");
  assert.equal(getEl("status-detail").classList.contains("error"), true);

  // Verify retry deadline formatted as local date/time, state, and attempt in textContent
  const expectedDateStr = new Date(futureDeadline).toLocaleString();
  const jobText = getEl("job-status").textContent;
  assert(jobText.includes(expectedDateStr), `job-status should contain local date/time "${expectedDateStr}", got: "${jobText}"`);
  assert(jobText.includes("Cooldown"), `job-status should include state "Cooldown", got: "${jobText}"`);
  assert(jobText.includes("attempt 2"), `job-status should include attempt "attempt 2", got: "${jobText}"`);

  // Verify manual sync button is disabled while deadline is in the future
  assert.equal(getEl("sync").disabled, true, "sync button must be disabled when future deadline exists");
}

// 2. Pause remains usable and sends automatic false; preserves cooldown while paused.
{
  const futureDeadline = Date.now() + 180000;
  let bgStatus = {
    title: "Rate limit cooldown",
    detail: "Waiting for backoff window (source: chatgpt-sync)",
    error: true,
  };
  let bgJob = {
    enabled: true,
    state: "cooldown",
    nextAttemptAt: futureDeadline,
    attempt: 1,
  };
  let bgSettings = {
    endpoint: "http://127.0.0.1:8765",
    token: "tok",
    automatic: true,
  };
  let recordedAutomaticPayload = null;

  const handler = async (msg) => {
    if (msg.action === "status") {
      return {
        status: bgStatus,
        settings: bgSettings,
        running: false,
        job: bgJob,
      };
    }
    if (msg.action === "automatic") {
      recordedAutomaticPayload = msg;
      bgSettings.automatic = msg.enabled;
      if (!msg.enabled) {
        bgJob = {
          enabled: false,
          state: "paused",
          nextAttemptAt: futureDeadline, // preserved cooldown while paused
          attempt: 1,
        };
      }
      return { ok: true };
    }
    return { ok: true };
  };

  const { getEl } = await setupPopup(handler);

  // Manual sync is disabled due to future cooldown deadline
  assert.equal(getEl("sync").disabled, true);
  // Checkbox is NOT disabled: usable while cooling down
  assert.equal(getEl("automatic").disabled, false);
  assert.equal(getEl("automatic").checked, true);

  // Pause the import
  getEl("automatic").checked = false;
  await getEl("automatic").dispatchEvent("change");

  // Verify { action: "automatic", enabled: false } sent
  assert.deepEqual(recordedAutomaticPayload, { action: "automatic", enabled: false });
  assert.equal(getEl("automatic").checked, false);

  // Paused state and preserved retry deadline displayed
  const pausedJobText = getEl("job-status").textContent;
  assert(pausedJobText.includes("Paused"));
  assert(pausedJobText.includes(new Date(futureDeadline).toLocaleString()));
  // Manual sync button remains disabled because nextAttemptAt is still in the future
  assert.equal(getEl("sync").disabled, true);
}

// 3. Normal refresh reflects completion unchecked; unsaved fields not overwritten; manual button re-enabled.
{
  let bgStatus = {
    title: "Importing history",
    detail: "Processing conversation page (source: chatgpt-web)",
    error: false,
  };
  let bgJob = {
    enabled: true,
    state: "running",
    nextAttemptAt: null,
    attempt: 1,
  };
  let bgSettings = {
    endpoint: "http://127.0.0.1:8765",
    token: "saved-token",
    workspaceId: "saved-ws",
    automatic: true,
  };

  const handler = async (msg) => {
    if (msg.action === "status") {
      return {
        status: bgStatus,
        settings: bgSettings,
        running: bgJob.state === "running",
        job: bgJob,
      };
    }
    return { ok: true };
  };

  const { getEl, intervals } = await setupPopup(handler);

  assert.equal(getEl("automatic").checked, true);
  assert.equal(getEl("sync").disabled, true);

  // User edits endpoint/token without saving yet
  getEl("endpoint").value = "http://127.0.0.1:9999/unsaved";
  getEl("token").value = "unsaved-token";
  getEl("workspace-id").value = "unsaved-ws";

  // Background finishes import: backfill catches up, automatic automatically becomes false
  bgJob = {
    enabled: false,
    state: "complete",
    nextAttemptAt: null,
    attempt: 0,
  };
  bgSettings.automatic = false;
  bgStatus = {
    title: "Import complete",
    detail: "All conversations indexed up to checkpoint.",
    error: false,
  };

  // Trigger routine periodic refresh tick
  assert(intervals.length > 0, "Periodic refresh interval must be registered");
  await intervals[0].fn();

  // Automatic checkbox updated to unchecked on routine refresh
  assert.equal(getEl("automatic").checked, false, "Routine refresh must uncheck automatic when backfill completes");
  // Sync button is now enabled
  assert.equal(getEl("sync").disabled, false);
  // Status updated
  assert.equal(getEl("status-title").textContent, "Import complete");
  assert(getEl("job-status").textContent.includes("Import complete"));

  // Crucial: unsaved fields MUST NOT have been overwritten during routine refresh
  assert.equal(getEl("endpoint").value, "http://127.0.0.1:9999/unsaved");
  assert.equal(getEl("token").value, "unsaved-token");
  assert.equal(getEl("workspace-id").value, "unsaved-ws");
}

// 4. Missing/no-response still shows actionable error.
{
  let failMessaging = false;
  const handler = async (msg) => {
    if (failMessaging) {
      return null; // No response from background
    }
    return {
      status: { title: "Ready", detail: "Paired." },
      settings: { endpoint: "http://127.0.0.1:8765", token: "tok", automatic: false },
      running: false,
      job: { state: "waiting", nextAttemptAt: null, attempt: 0 },
    };
  };

  const { getEl } = await setupPopup(handler);
  assert.equal(getEl("status-title").textContent, "Ready");

  // Background bridge drops
  failMessaging = true;
  await getEl("sync").dispatchEvent("click");

  assert.equal(getEl("status-title").textContent, "Needs attention");
  assert.equal(getEl("status-detail").textContent, "The background bridge did not respond.");
  assert.equal(getEl("status-detail").classList.contains("error"), true);
}

// 5. Messages are inert text; safe against HTML injection.
{
  const xssPayload = '<img src=x onerror="alert(1)"> & <script>alert("xss")</script>';
  const handler = async (msg) => {
    return {
      status: {
        title: xssPayload,
        detail: xssPayload,
        error: true,
      },
      settings: { endpoint: "http://127.0.0.1:8765", token: "tok", automatic: false },
      running: false,
      job: {
        state: "cooldown",
        nextAttemptAt: Date.now() + 60000,
        attempt: 1,
      },
    };
  };

  const { getEl } = await setupPopup(handler);

  // Literal text stored inertly
  assert.equal(getEl("status-title").textContent, xssPayload);
  assert.equal(getEl("status-detail").textContent, xssPayload);
  assert.equal(getEl("status-title")._classes.size, 0);
  assert(getEl("job-status").textContent.includes("Cooldown (attempt 1)"));
}

// 6. Actionable error bounded in length for save failure.
{
  const longErrorMsg = "F".repeat(400);
  const handler = async (msg) => {
    if (msg.action === "status") {
      return {
        status: { title: "Ready", detail: "OK" },
        settings: { endpoint: "http://127.0.0.1:8765", token: "tok", automatic: false },
        running: false,
        job: null,
      };
    }
    if (msg.action === "save") {
      return { ok: false, error: longErrorMsg };
    }
    return { ok: true };
  };

  const { getEl } = await setupPopup(handler);
  await getEl("settings-form").dispatchEvent("submit");

  assert.equal(getEl("status-title").textContent, "Needs attention");
  assert.equal(getEl("status-detail").textContent.length, 240);
  assert.equal(getEl("status-detail").textContent, "F".repeat(240));
}

console.log("Browser extension popup offline checks passed (synthetic VM, queued 429 diagnostic preservation, future deadline sync disable, pause control, completion update, transport error, inert text).");
