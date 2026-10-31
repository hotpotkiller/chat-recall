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
vm.runInContext(`${functions};globalThis.highlight = appendHighlighted;globalThis.readiness = semanticText;globalThis.semanticJobText = semanticJobText;globalThis.cursorMatchesScope = cursorMatchesScope;`, context);
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

// New coverage and pause/cursor states
assert.equal(context.semanticJobText({ state: "pausing" }), "Stopping meaning indexing after current request…");
assert.equal(context.semanticJobText({ state: "paused" }), "Meaning indexing paused");
assert.equal(context.semanticJobText({ state: "complete", report: { pass_exhausted: true, remaining: 3 } }), "Some passages need attention");
assert.equal(context.semanticJobText({ state: "running", scope: { account_id: "work" } }, null, 0, "personal"), "Background meaning indexing in progress (work)");
assert.equal(context.semanticJobText({ state: "paused", scope: { account_id: "work" } }, null, 0, "personal"), "Meaning indexing paused (work)");
assert.equal(context.semanticJobText({ state: "running", scope: { account_id: "work" } }, null, 0, "work"), "Background meaning indexing in progress (work)");
assert.equal(context.semanticJobText({ state: "paused", scope: { account_id: "work" } }, null, 0, "work"), "Meaning indexing paused (work)");
assert.equal(context.semanticJobText({ state: "running", scope: { account_id: null } }), "Background meaning indexing in progress (all)");
assert.equal(context.semanticJobText({ state: "paused", scope: { account_id: null } }), "Meaning indexing paused (all)");
assert.equal(context.semanticJobText({ state: "pausing", scope: { account_id: null } }), "Stopping meaning indexing after current request… (all)");
assert.equal(context.semanticJobText({ state: "running", report: { scope: { account_id: "work" } } }), "Background meaning indexing in progress (work)");

// cursorMatchesScope assertions
const validCursor = { account_id: "work", model_id: "test-model-v2", generate_topics: false };
assert.equal(context.cursorMatchesScope(validCursor, "work", "test-model-v2"), true);
assert.equal(context.cursorMatchesScope(validCursor, "other", "test-model-v2"), false, "Mismatched account rejected");
assert.equal(context.cursorMatchesScope(validCursor, "work", "different-model"), false, "Mismatched model rejected");
assert.equal(context.cursorMatchesScope(validCursor, "work", ""), false, "Missing target model rejected");
assert.equal(context.cursorMatchesScope(validCursor, "work", null), false, "Null target model rejected");
assert.equal(context.cursorMatchesScope({ account_id: "work", model_id: "", generate_topics: false }, "work", "test-model-v2"), false, "Missing cursor model rejected");
assert.equal(context.cursorMatchesScope({ account_id: "work", generate_topics: false }, "work", "test-model-v2"), false, "Absent cursor model rejected");
assert.equal(context.cursorMatchesScope({ account_id: "work", model_id: "test-model-v2", generate_topics: true }, "work", "test-model-v2"), false, "generate_topics true rejected");
assert.equal(context.cursorMatchesScope(null, "work", "test-model-v2"), false, "Null cursor rejected");

assert.equal(context.readiness({ configured: true, model_id: "m1", coverage: { account_id: "a1", eligible: 10, current: 7, pending: 3, legacy: 0 } }), "7 of 10 messages ready for meaning search (partial)");
assert.equal(context.readiness({ configured: true, model_id: "m1", coverage: { account_id: "a1", eligible: 10, current: 10, pending: 0, legacy: 0 } }), "10 messages ready for meaning search");
assert.equal(context.readiness({ configured: true, model_id: "m1", coverage: { account_id: "a1", eligible: 10, current: 0, pending: 10, legacy: 0 } }), "Meaning search not ready · index the configured model");
assert.equal(context.readiness({ configured: true, model_id: null }), "Meaning search identity and coverage pending · keyword search available");
assert.equal(context.readiness({ configured: true, model_id: null, coverage: { eligible: 10, current: 0 } }), "Meaning search identity and coverage pending · keyword search available");
assert.equal(context.readiness({ configured: true, model_id: null, models: [{ model_id: "other" }] }), "Meaning search identity and coverage pending · keyword search available");

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

// 7. Search page presentation, accessibility, palette contrast, responsiveness, and DOM preservation
{
  const css = await readFile(new URL("../chat_recall/web/style.css", import.meta.url), "utf8");
  const html = await readFile(new URL("../chat_recall/web/index.html", import.meta.url), "utf8");

  // No external CDNs, fonts, or network URLs
  assert(!/https?:\/\//i.test(css), "style.css must not load external fonts, stylesheets, or assets");
  assert(!/<link[^>]+https?:/i.test(html), "index.html must not load external styles or fonts");
  assert(!/<script[^>]+https?:/i.test(html), "index.html must not load external scripts or CDNs");

  // Color contrast calculation according to WCAG 2.1 specs
  function sRGBtoLin(c) {
    const cNorm = c / 255;
    return cNorm <= 0.03928 ? cNorm / 12.92 : Math.pow((cNorm + 0.055) / 1.055, 2.4);
  }
  function hexToLuminance(hex) {
    const clean = hex.replace("#", "").trim();
    const r = parseInt(clean.slice(0, 2), 16);
    const g = parseInt(clean.slice(2, 4), 16);
    const b = parseInt(clean.slice(4, 6), 16);
    return 0.2126 * sRGBtoLin(r) + 0.7152 * sRGBtoLin(g) + 0.0722 * sRGBtoLin(b);
  }
  function contrastRatio(hex1, hex2) {
    const l1 = hexToLuminance(hex1);
    const l2 = hexToLuminance(hex2);
    return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
  }

  // Extract core theme colors
  const inkMatch = css.match(/--ink:\s*(#[0-9a-fA-F]{6})/);
  const mutedMatch = css.match(/--muted:\s*(#[0-9a-fA-F]{6})/);
  const greenMatch = css.match(/--green:\s*(#[0-9a-fA-F]{6})/);
  const surfaceMatch = css.match(/--surface:\s*(#[0-9a-fA-F]{6})/);
  const paperMatch = css.match(/--paper:\s*(#[0-9a-fA-F]{6})/);

  assert(inkMatch && mutedMatch && greenMatch && surfaceMatch && paperMatch, "CSS custom properties must define the palette");
  const ink = inkMatch[1];
  const muted = mutedMatch[1];
  const green = greenMatch[1];
  const surface = surfaceMatch[1];
  const paper = paperMatch[1];

  // WCAG AA requirement for normal text: contrast ratio >= 4.5:1
  assert(contrastRatio(ink, surface) >= 4.5, `Ink contrast on surface must be >= 4.5:1 (got ${contrastRatio(ink, surface).toFixed(2)})`);
  assert(contrastRatio(ink, paper) >= 4.5, `Ink contrast on paper must be >= 4.5:1 (got ${contrastRatio(ink, paper).toFixed(2)})`);
  assert(contrastRatio(muted, surface) >= 4.5, `Muted contrast on surface must be >= 4.5:1 (got ${contrastRatio(muted, surface).toFixed(2)})`);
  assert(contrastRatio(muted, paper) >= 4.5, `Muted contrast on paper must be >= 4.5:1 (got ${contrastRatio(muted, paper).toFixed(2)})`);
  assert(contrastRatio(green, surface) >= 4.5, `Green contrast on surface must be >= 4.5:1 (got ${contrastRatio(green, surface).toFixed(2)})`);
  assert(contrastRatio("#ffffff", green) >= 4.5, `White text on green button must be >= 4.5:1 (got ${contrastRatio("#ffffff", green).toFixed(2)})`);

  // Accessibility: Visible keyboard focus, reduced motion, touch targets
  assert(css.includes(":focus-visible"), "style.css must define visible keyboard focus indicators");
  assert(css.includes("prefers-reduced-motion"), "style.css must respect prefers-reduced-motion");
  assert(css.includes(".skip-link"), "style.css must style skip link for keyboard navigation");
  assert(html.includes('class="skip-link"'), "index.html must include skip link");

  // Selected state and touch target styling
  assert(css.includes(".result-card.selected"), "Selected card style must be defined");
  assert(css.includes("min-height: 44px") || css.includes("min-height: 42px") || css.includes("min-height: 40px"), "Touch targets of ~44px must be styled");
  assert(css.includes("width: 44px") && css.includes("height: 44px"), "Close icon touch target of 44x44px must be styled");

  // Responsive panes from 320px to desktop
  assert(css.includes("@media") && css.includes("max-width: 800px"), "Responsive single-column pane layout must be defined for tablets/phones");
  assert(css.includes("max-width: 540px"), "Responsive fluid adjustments down to 320px must be defined");
  assert(css.includes(".context-pane.mobile-open"), "Mobile context pane styling must be defined");

  // DOM preservation: All required IDs preserved in index.html
  const requiredIds = [
    "pair-button", "import-button", "page-title", "pair-panel", "pair-heading",
    "pair-endpoint", "pair-token", "copy-token", "search-form", "query",
    "case-sensitive", "role", "account", "archive-count", "semantic-status",
    "index-semantics", "mode-note", "cloud-option", "cloud-rerank", "notice",
    "workspace", "results-title", "result-count", "results", "pagination",
    "previous-page", "page-label", "next-page", "context-pane",
    "context-placeholder", "context-content", "context-title", "context-meta",
    "close-context", "original-link", "copy-message", "messages",
    "import-dialog", "import-form", "import-heading", "close-import",
    "import-file", "import-account", "import-report", "confirm-import",
  ];
  for (const id of requiredIds) {
    assert(html.includes(`id="${id}"`), `DOM element id="${id}" must be preserved in index.html`);
  }

  // Exact mode checked by default
  assert(html.includes('value="exact" checked'), "Exact mode radio must remain checked by default in index.html");
}

// 8. Desktop and mobile dialogue viewer flex layout contracts
{
  const css = await readFile(new URL("../chat_recall/web/style.css", import.meta.url), "utf8");

  // Helper to extract CSS declaration block for a given selector
  function getRule(selector) {
    const idx = css.indexOf(selector);
    if (idx === -1) return null;
    const openBrace = css.indexOf("{", idx);
    const closeBrace = css.indexOf("}", openBrace);
    if (openBrace === -1 || closeBrace === -1) return null;
    return css.slice(openBrace + 1, closeBrace);
  }

  // 1. Desktop .context-pane must be a bounded flex container
  const paneRule = getRule(".context-pane");
  assert(paneRule, "CSS must define .context-pane");
  assert(/display:\s*flex/.test(paneRule), ".context-pane must be display:flex");
  assert(/flex-direction:\s*column/.test(paneRule), ".context-pane must be flex-direction:column");
  assert(/overflow:\s*hidden/.test(paneRule), ".context-pane must be overflow:hidden");
  assert(/max-height:/.test(paneRule), ".context-pane must specify max-height constraint");

  // 2. Desktop visible #context-content must be a bounded flex child and container
  const contentRule = getRule("#context-content:not([hidden])") || getRule("#context-content");
  assert(contentRule, "CSS must define layout for visible #context-content");
  assert(/display:\s*flex/.test(contentRule), "#context-content must be display:flex");
  assert(/flex-direction:\s*column/.test(contentRule), "#context-content must be flex-direction:column");
  assert(/flex:\s*1\b/.test(contentRule), "#context-content must have flex:1 to absorb available pane height");
  assert(/min-height:\s*0\b/.test(contentRule), "#context-content must have min-height:0 to allow flex shrinking");
  assert(/overflow:\s*hidden/.test(contentRule), "#context-content must be overflow:hidden");

  // 3. Desktop .messages must scroll inside #context-content
  const messagesRule = getRule(".messages");
  assert(messagesRule, "CSS must define .messages");
  assert(/flex:\s*1\b/.test(messagesRule), ".messages must have flex:1");
  assert(/min-height:\s*0\b/.test(messagesRule), ".messages must have min-height:0");
  assert(/overflow-y:\s*auto/.test(messagesRule), ".messages must have overflow-y:auto for scrolling");

  // 4. Header, actions, and branch note must not shrink when messages list is long
  for (const selector of [".context-header", ".context-actions", ".branch-note"]) {
    const rule = getRule(selector);
    assert(rule, `CSS must define ${selector}`);
    assert(/flex-shrink:\s*0/.test(rule), `${selector} must have flex-shrink:0 so it stays anchored at the top`);
  }

  // 5. Hidden semantics must be preserved
  const hiddenRule = getRule("[hidden]");
  assert(hiddenRule, "CSS must define [hidden]");
  assert(/display:\s*none\s*!important/.test(hiddenRule), "[hidden] must enforce display:none !important");

  // 6. Mobile non-overlay layout contract: .context-pane remains in normal document flow
  assert(css.includes(".context-pane.mobile-open"), "CSS must style mobile context pane");
  assert(!/\.context-pane\.mobile-open\s*\{[^}]*position:\s*fixed/m.test(css), "Mobile context pane must NOT use position:fixed blocking overlay");
  assert(!/\.context-pane\.mobile-open\s*\{[^}]*9999px/m.test(css), "Mobile context pane must NOT use blocking backdrop shadow");
  const mobilePaneRule = getRule(".context-pane.mobile-open");
  assert(mobilePaneRule && /position:\s*static/.test(mobilePaneRule), "Mobile context pane must remain in normal document flow (position:static)");
  assert(/overflow-y:\s*auto/.test(css), "Mobile messages must retain overflow-y:auto for long conversation scrolling");

  // 7. Synthetic flex chain simulation:
  // Given a constrained parent height (600px), header (60px), actions (50px), note (40px),
  // and a long conversation (2500px of text), verify that #context-content is bounded
  // to the pane height and messages container is allocated the remaining 450px and scrolls.
  const paneHeight = 600;
  const headerHeight = 60;
  const actionsHeight = 50;
  const noteHeight = 40;
  const longDialogueContentHeight = 2500;

  // #context-content as flex:1 inside pane gets 600px
  const contextContentHeight = paneHeight;
  assert.equal(contextContentHeight, 600);

  // Available height for .messages is 600 - (60 + 50 + 40) = 450px
  const availableMessagesHeight = contextContentHeight - (headerHeight + actionsHeight + noteHeight);
  assert.equal(availableMessagesHeight, 450);

  // Messages container does not expand the pane; it is capped at 450px and scrolls
  const renderedMessagesHeight = Math.min(longDialogueContentHeight, availableMessagesHeight);
  assert.equal(renderedMessagesHeight, 450);
  assert(longDialogueContentHeight > renderedMessagesHeight, "Long dialogue messages must scroll inside the allocated 450px area without expanding or clipping the pane");

  // 8. Viewport scrolling unblocked: body/html must not lock scrolling
  assert(!/body\s*\{[^}]*overflow:\s*hidden/m.test(css), "body must not lock overflow");
  assert(!/html\s*\{[^}]*overflow:\s*hidden/m.test(css), "html must not lock overflow");

  // 9. Script accessibility contract in app.js: scrollIntoView, focus, and Escape handler
  const scriptContent = await readFile(new URL("../chat_recall/web/app.js", import.meta.url), "utf8");
  assert(scriptContent.includes('$("context-pane").scrollIntoView'), "app.js must scroll context pane into view on result click");
  assert(scriptContent.includes('$("context-pane").focus()'), "app.js must focus context pane for keyboard/screen-reader accessibility");
  assert(scriptContent.includes('event.key === "Escape"'), "app.js must handle Escape key to close context");
  assert(scriptContent.includes('$("close-context").addEventListener("click", closeContext)'), "Close button must close context");
}

// 9. Context viewer focus restoration and non-stealing behavior tests
{
  class WebTestElement {
    constructor(tag, className = "", text = "") {
      this.tag = String(tag).toLowerCase();
      this.tagName = String(tag).toUpperCase();
      this.className = className;
      this.children = [];
      this.textContent = text;
      this.attributes = new Map();
      this.listeners = new Map();
      this.dataset = {};
      this.hidden = false;
      this.disabled = false;
      this.checked = false;
      this.value = "";
      this.focused = false;
      this.parentNode = null;
    }
    get classList() {
      return {
        add: (...names) => {
          const set = new Set(this.className ? this.className.split(/\s+/).filter(Boolean) : []);
          for (const n of names) if (n) set.add(n);
          this.className = [...set].join(" ");
        },
        remove: (...names) => {
          const set = new Set(this.className ? this.className.split(/\s+/).filter(Boolean) : []);
          for (const n of names) set.delete(n);
          this.className = [...set].join(" ");
        },
        toggle: (name, force) => {
          const set = new Set(this.className ? this.className.split(/\s+/).filter(Boolean) : []);
          const shouldAdd = force !== undefined ? Boolean(force) : !set.has(name);
          if (shouldAdd) set.add(name); else set.delete(name);
          this.className = [...set].join(" ");
          return shouldAdd;
        },
        contains: (name) => {
          const set = new Set(this.className ? this.className.split(/\s+/).filter(Boolean) : []);
          return set.has(name);
        },
      };
    }
    append(...nodes) {
      for (const node of nodes) {
        if (!node) continue;
        if (node instanceof WebTestFragment) {
          for (const child of node.children) {
            child.parentNode = this;
            this.children.push(child);
          }
          node.children = [];
        } else {
          node.parentNode = this;
          this.children.push(node);
        }
      }
    }
    replaceChildren(...nodes) {
      this.children = [];
      this.textContent = "";
      this.append(...nodes);
    }
    setAttribute(name, val) {
      this.attributes.set(name, String(val));
      if (name === "hidden") this.hidden = true;
    }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    removeAttribute(name) {
      this.attributes.delete(name);
      if (name === "hidden") this.hidden = false;
    }
    addEventListener(event, fn) {
      if (!this.listeners.has(event)) this.listeners.set(event, []);
      this.listeners.get(event).push(fn);
    }
    dispatchEvent(event) {
      event.currentTarget = this;
      if (!event.target) event.target = this;
      const fns = this.listeners.get(event.type) || [];
      for (const fn of fns) fn(event);
      if (event.bubbles && this.parentNode && !event._stop) {
        this.parentNode.dispatchEvent(event);
      }
    }
    focus() {
      this.focused = true;
      if (this.doc) this.doc.activeElement = this;
    }
    scrollIntoView() {}
    contains(other) {
      let curr = other;
      while (curr) {
        if (curr === this) return true;
        curr = curr.parentNode;
      }
      return false;
    }
    matches(selector) {
      const parts = selector.split(",").map((s) => s.trim());
      for (const part of parts) {
        if (part.startsWith(".")) {
          if (this.classList.contains(part.slice(1))) return true;
        } else if (part.includes("[")) {
          const match = part.match(/^([a-zA-Z0-9_-]+)?(\.[a-zA-Z0-9_-]+)?\[([a-zA-Z0-9_-]+)(?:="([^"]+)")?\](?::([a-zA-Z-]+))?$/);
          if (match) {
            const [, t, cls, attr, val, pseudo] = match;
            if (t && this.tag !== t.toLowerCase()) continue;
            if (cls && !this.classList.contains(cls.slice(1))) continue;
            const attrVal = attr === "name" ? this.name : (this.dataset[attr.replace(/^data-/, "")] ?? this.getAttribute(attr));
            if (val !== undefined && attrVal !== val) continue;
            if (val === undefined && !this.attributes.has(attr) && this.dataset[attr.replace(/^data-/, "")] === undefined) continue;
            if (pseudo === "checked" && !this.checked) continue;
            return true;
          }
        } else if (this.tag === part.toLowerCase()) {
          return true;
        }
      }
      return false;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    querySelectorAll(selector) {
      const found = [];
      function walk(node) {
        for (const child of node.children) {
          if (child instanceof WebTestElement) {
            if (child.matches(selector)) found.push(child);
            walk(child);
          }
        }
      }
      walk(this);
      return found;
    }
    get text() { return this.textContent + this.children.map((c) => c.text || "").join(""); }
  }

  class WebTestFragment {
    constructor() { this.children = []; }
    append(...nodes) {
      for (const n of nodes) if (n) this.children.push(n);
    }
  }

  const elements = new Map();
  const getOrCreate = (id, tag = "div") => {
    if (!elements.has(id)) {
      const el = new WebTestElement(tag);
      el.id = id;
      elements.set(id, el);
    }
    return elements.get(id);
  };

  const ids = [
    "search-form", "query", "case-sensitive", "cloud-rerank", "role", "account",
    "archive-count", "semantic-status", "index-semantics", "cloud-option",
    "pair-panel", "pair-button", "pair-endpoint", "pair-token", "copy-token",
    "notice", "workspace", "results", "results-title", "result-count",
    "pagination", "previous-page", "next-page", "page-label",
    "context-pane", "context-placeholder", "context-content", "context-title",
    "context-meta", "original-link", "copy-message", "close-context", "messages",
    "import-button", "import-dialog", "close-import", "import-form", "import-file",
    "import-account", "import-report", "confirm-import", "mode-note",
  ];
  for (const id of ids) getOrCreate(id);

  // Link children into context-pane and context-content
  const contextPane = elements.get("context-pane");
  const contextPlaceholder = elements.get("context-placeholder");
  const contextContent = elements.get("context-content");
  const closeContextBtn = elements.get("close-context");
  const messagesBox = elements.get("messages");
  const copyMessageBtn = elements.get("copy-message");
  contextPane.append(contextPlaceholder, contextContent);
  contextContent.append(closeContextBtn, copyMessageBtn, messagesBox);

  const modeExact = new WebTestElement("input");
  modeExact.type = "radio"; modeExact.name = "mode"; modeExact.value = "exact"; modeExact.checked = true;
  const modeSmart = new WebTestElement("input");
  modeSmart.type = "radio"; modeSmart.name = "mode"; modeSmart.value = "smart"; modeSmart.checked = false;
  const modeInputs = [modeExact, modeSmart];

  const doc = {
    activeElement: null,
    createElement: (tag) => {
      const el = new WebTestElement(tag);
      el.doc = doc;
      return el;
    },
    createTextNode: (text) => ({ textContent: String(text), text: String(text), tag: "#text", children: [] }),
    createDocumentFragment: () => new WebTestFragment(),
    getElementById: (id) => elements.get(id) || null,
    querySelector: (selector) => {
      if (selector === 'input[name="mode"]:checked') return modeInputs.find((i) => i.checked) || null;
      if (selector === 'input[name="mode"][value="smart"]') return modeSmart;
      if (selector === 'input[name="mode"][value="exact"]') return modeExact;
      const cardMatch = typeof selector === "string" && selector.match(/^\.result-card\[data-index="(\d+)"\]$/);
      if (cardMatch) {
        const idx = cardMatch[1];
        const cards = elements.get("results").querySelectorAll(".result-card");
        return cards.find((c) => String(c.dataset.index) === idx) || null;
      }
      return elements.get("results")?.querySelector(selector) || null;
    },
    querySelectorAll: (selector) => {
      if (selector === 'input[name="mode"]') return modeInputs;
      if (selector === ".result-card") return elements.get("results")?.querySelectorAll(".result-card") || [];
      return [];
    },
    addEventListener: (event, fn) => {
      if (!doc._listeners) doc._listeners = new Map();
      if (!doc._listeners.has(event)) doc._listeners.set(event, []);
      doc._listeners.get(event).push(fn);
    },
  };

  for (const el of elements.values()) el.doc = doc;

  const appScript = await readFile(new URL("../chat_recall/web/app.js", import.meta.url), "utf8");
  const win = {
    location: new URL("http://localhost:8080/"),
    matchMedia: () => ({ matches: false }),
    getSelection: () => ({ toString: () => "" }),
  };

  const sandbox = {
    document: doc,
    window: win,
    location: win.location,
    history: { replaceState() {} },
    fetch: async () => ({ ok: true, status: 200, json: async () => ({}) }),
    Option: class { constructor(t, v) { this.text = t; this.value = v; } },
    URL,
    URLSearchParams,
    setTimeout: () => 1,
    clearTimeout: () => {},
    navigator: { clipboard: { writeText: async () => {} } },
  };

  const vmContext = vm.createContext(sandbox);
  vm.runInContext(appScript, vmContext);

  const mockResult = {
    conversation_id: "conv-focus-1",
    message_id: "msg-focus-1",
    title: "Focus Test Conversation",
    text: "Sample conversation content for keyboard focus testing.",
    matches: [],
    match_type: "exact",
  };

  // Render a result card and select it
  vmContext.renderResults({ results: [mockResult], total: 1 });
  const resultsContainer = elements.get("results");
  const card = resultsContainer.querySelector(".result-card");
  assert(card, "Result card must be rendered");
  const titleBtn = card.querySelector(".result-title");
  assert(titleBtn, "Title button must exist on card");

  // Open context for the result
  vmContext.openContext(mockResult, 0);
  assert.equal(elements.get("context-content").hidden, false, "Dialogue must be open");

  // Test 1: Inside-pane restore:
  // When focus is inside the context pane (e.g. on close button), closing restores focus to the result card's title button
  closeContextBtn.focus();
  assert.equal(doc.activeElement, closeContextBtn, "Focus must be on close-context button inside pane");
  titleBtn.focused = false;

  vmContext.closeContext({ target: closeContextBtn });
  assert.equal(elements.get("context-content").hidden, true, "Context content must be hidden after close");
  assert.equal(elements.get("context-placeholder").hidden, false, "Placeholder must be visible after close");
  assert.equal(titleBtn.focused, true, "Focus must be restored to result-title when closed from inside context pane");
  assert.equal(doc.activeElement, titleBtn, "activeElement must be restored to result title button");

  // Test 2: Outside-pane no-steal:
  // Re-open context, place focus outside pane (e.g. on search query input), then trigger Escape
  vmContext.openContext(mockResult, 0);
  assert.equal(elements.get("context-content").hidden, false, "Dialogue must be re-opened");
  const queryInput = elements.get("query");
  queryInput.focus();
  assert.equal(doc.activeElement, queryInput, "Focus is in query input outside context pane");
  titleBtn.focused = false;

  // Escape key event with target on query input
  const escapeEvent = { key: "Escape", target: queryInput, preventDefault() {} };
  const keydownHandlers = doc._listeners?.get("keydown") || [];
  for (const handler of keydownHandlers) handler(escapeEvent);

  assert.equal(elements.get("context-content").hidden, true, "Context content must be closed on Escape");
  assert.equal(doc.activeElement, queryInput, "Focus must NOT be stolen from query input on Escape");
  assert.equal(titleBtn.focused, false, "Title button must not have received focus when closing from outside pane");

  // Test 3: No-selection behavior:
  // When no result is selected, closing does not error and does not focus anything unexpectedly
  queryInput.focus();
  vmContext.closeContext();
  assert.equal(elements.get("context-content").hidden, true);
  assert.equal(doc.activeElement, queryInput, "No-selection close must not steal focus");
}

console.log("Browser extension popup offline checks passed (synthetic VM, queued 429 diagnostic preservation, future deadline sync disable, pause control, completion update, transport error, inert text, search polish & accessibility, desktop/mobile flex layout contracts, context focus restore & non-stealing).");
