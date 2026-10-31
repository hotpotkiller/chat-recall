// Offline tests for Smart dialogue activation, visibility, focus, Unicode anchor,
// partial coverage reporting, and keyword fallback.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

class Element {
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
    this.files = [];
    this.tabIndex = -1;
    this.scrollTop = 0;
    this.offsetTop = 0;
    this.focused = false;
    this.scrolledIntoView = false;
    this.scrollOptions = null;
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
      if (node instanceof DocumentFragment) {
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

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

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
    for (const fn of fns) {
      if (event._stopImmediate) break;
      fn(event);
    }
    if (event.bubbles && this.parentNode && !event._stop) {
      this.parentNode.dispatchEvent(event);
    }
  }

  click() {
    const event = {
      type: "click",
      target: this,
      currentTarget: this,
      bubbles: true,
      defaultPrevented: false,
      _stop: false,
      _stopImmediate: false,
      preventDefault() { this.defaultPrevented = true; },
      stopPropagation() { this._stop = true; },
      stopImmediatePropagation() { this._stopImmediate = true; },
    };
    this.dispatchEvent(event);
  }

  focus() {
    this.focused = true;
  }

  scrollIntoView(options) {
    this.scrolledIntoView = true;
    this.scrollOptions = options;
  }

  closest(selector) {
    let curr = this;
    while (curr) {
      if (curr.matches && curr.matches(selector)) return curr;
      curr = curr.parentNode;
    }
    return null;
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

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }

  querySelectorAll(selector) {
    const found = [];
    function walk(node) {
      for (const child of node.children) {
        if (child instanceof Element) {
          if (child.matches(selector)) found.push(child);
          walk(child);
        }
      }
    }
    walk(this);
    return found;
  }

  get text() {
    return this.textContent + this.children.map((child) => child.text || "").join("");
  }
}

class TextNode {
  constructor(text = "") {
    this.textContent = String(text);
    this.tag = "#text";
    this.children = [];
    this.parentNode = null;
  }
  get text() { return this.textContent; }
}

class DocumentFragment {
  constructor() {
    this.children = [];
  }
  append(...nodes) {
    for (const node of nodes) {
      if (node) this.children.push(node);
    }
  }
}

class Option {
  constructor(text, value) {
    this.text = text;
    this.value = value;
  }
}

async function createTestEnv(options = {}) {
  const elements = new Map();
  function getOrCreate(id, tag = "div") {
    if (!elements.has(id)) {
      const el = new Element(tag);
      el.id = id;
      elements.set(id, el);
    }
    return elements.get(id);
  }

  const ids = [
    "search-form", "query", "case-sensitive", "cloud-rerank", "role", "account",
    "archive-count", "semantic-status", "index-semantics", "cloud-option",
    "pair-panel", "pair-button", "pair-endpoint", "pair-token", "copy-token",
    "notice", "workspace", "results", "results-title", "result-count",
    "pagination", "previous-page", "next-page", "page-label",
    "context-pane", "context-placeholder", "context-content", "context-title",
    "context-meta", "original-link", "copy-message", "close-context", "messages",
    "import-button", "import-dialog", "close-import", "import-form", "import-file",
    "import-account", "import-report", "confirm-import", "mode-note", "pause-enrichment",
  ];
  const hiddenInitialIds = new Set([
    "pair-panel", "index-semantics", "cloud-option", "notice", "pagination", "context-content", "pause-enrichment",
  ]);
  for (const id of ids) {
    const el = getOrCreate(id);
    if (hiddenInitialIds.has(id)) el.hidden = true;
  }

  const modeExact = new Element("input");
  modeExact.type = "radio";
  modeExact.name = "mode";
  modeExact.value = "exact";
  modeExact.checked = true;

  const modeSmart = new Element("input");
  modeSmart.type = "radio";
  modeSmart.name = "mode";
  modeSmart.value = "smart";
  modeSmart.checked = false;

  const modeInputs = [modeExact, modeSmart];

  let activeSelectionText = "";

  const doc = {
    createElement: (tag) => new Element(tag),
    createTextNode: (text) => new TextNode(text),
    createDocumentFragment: () => new DocumentFragment(),
    getElementById: (id) => elements.get(id) || null,
    querySelector: (selector) => {
      if (selector === 'input[name="mode"]:checked') return modeInputs.find((i) => i.checked) || null;
      if (selector === 'input[name="mode"][value="smart"]') return modeSmart;
      if (selector === 'input[name="mode"][value="exact"]') return modeExact;
      const cardMatch = typeof selector === "string" && selector.match(/^\.result-card\[data-index="(\d+)"\]$/);
      if (cardMatch) {
        const idx = cardMatch[1];
        const resultsEl = elements.get("results");
        if (resultsEl) {
          const directCards = (resultsEl.children || []).filter((c) => c.classList && c.classList.contains("result-card"));
          const match = directCards.find((c) => String(c.dataset && c.dataset.index) === idx);
          if (match) return match;
          const queryCards = resultsEl.querySelectorAll(".result-card");
          return queryCards.find((c) => String(c.dataset && c.dataset.index) === idx) || null;
        }
        return null;
      }
      return elements.get("results")?.querySelector(selector) || null;
    },
    querySelectorAll: (selector) => {
      if (selector === 'input[name="mode"]') return modeInputs;
      if (selector === ".result-card") {
        return elements.get("results")?.querySelectorAll(".result-card") || [];
      }
      return [];
    },
    addEventListener: () => {},
  };

  const fetchCalls = [];
  let contextFetchHandler = null;
  let statusFetchHandler = typeof options.status === "function" ? options.status : (options.status ? (() => options.status) : null);

  async function mockFetch(url, fetchOpts = {}) {
    const urlStr = String(url);
    fetchCalls.push({ url: urlStr, options: fetchOpts });
    if (urlStr === "/api/config") {
      const configData = {
        token: "synthetic-token",
        semantic_available: true,
        cloud_rerank_available: false,
        ...(options.config || {}),
      };
      return { ok: true, status: 200, json: async () => configData };
    }
    if (urlStr.startsWith("/api/status")) {
      if (statusFetchHandler) {
        const handlerResult = await statusFetchHandler(urlStr, fetchOpts);
        return { ok: true, status: 200, json: async () => handlerResult };
      }
      const statusData = typeof options.status === "function"
        ? await options.status(urlStr, fetchOpts)
        : {
            conversations: 1,
            messages: 10,
            sources: [],
            ...(typeof options.status === "object" ? options.status : {}),
          };
      return { ok: true, status: 200, json: async () => statusData };
    }
    if (urlStr.startsWith("/api/context")) {
      if (contextFetchHandler) return contextFetchHandler(urlStr, fetchOpts);
      return {
        ok: true,
        status: 200,
        json: async () => ({
          title: "Synthetic Conversation",
          url: "https://chatgpt.com/c/synthetic-id",
          messages: [
            { id: "m1", role: "user", text: "synthetic question" },
            { id: "m2", role: "assistant", text: "synthetic reply" },
          ],
        }),
      };
    }
    if (urlStr.startsWith("/api/enrich/pause")) {
      if (typeof options.status === "object" && options.status?.semantic_job) {
        options.status.semantic_job.state = "pausing";
      }
      const prevHandler = statusFetchHandler;
      if (prevHandler) {
        statusFetchHandler = async (u, f) => {
          const res = await prevHandler(u, f);
          if (res && res.semantic_job) {
            return { ...res, semantic_job: { ...res.semantic_job, state: "pausing" } };
          }
          return res;
        };
      }
      return { ok: true, status: 200, json: async () => ({ status: "pausing" }) };
    }
    if (urlStr.startsWith("/api/enrich")) {
      return { ok: true, status: 200, json: async () => ({ enriched: 0 }) };
    }
    if (urlStr.startsWith("/api/search")) {
      return { ok: true, status: 200, json: async () => ({ results: [], total: 0 }) };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  }

  const script = await readFile(new URL("../chat_recall/web/app.js", import.meta.url), "utf8");
  const loc = typeof options.location === "string"
    ? new URL(options.location)
    : (options.location || new URL("http://localhost:8080/"));
  const win = {
    getSelection: () => ({ toString: () => activeSelectionText }),
    location: loc,
  };

  const historyCalls = [];
  const scheduledTimeouts = [];
  const sandbox = {
    document: doc,
    window: win,
    location: win.location,
    history: {
      replaceState(stateObj, title, url) {
        historyCalls.push({ state: stateObj, title, url: String(url) });
        if (url) {
          try {
            const parsed = new URL(url, win.location.href);
            win.location.href = parsed.href;
            win.location.search = parsed.search;
            win.location.pathname = parsed.pathname;
          } catch {}
        }
      },
    },
    fetch: mockFetch,
    Option,
    URL,
    URLSearchParams,
    setTimeout: (fn, delay = 0) => {
      const timerDelay = delay >= 1000 ? delay : 0;
      const id = setTimeout(fn, timerDelay);
      if (typeof id?.unref === "function") id.unref();
      scheduledTimeouts.push({ fn, delay, id });
      return id;
    },
    clearTimeout: (id) => {
      clearTimeout(id);
    },
    navigator: { clipboard: { writeText: async () => {} } },
  };

  const vmContext = vm.createContext(sandbox);
  vm.runInContext(script, vmContext);
  // Wait for start() microtasks, async config/status, and initial searches to complete
  for (let i = 0; i < 25; i++) {
    await new Promise((r) => setImmediate(r));
  }

  return {
    vmContext,
    elements,
    doc,
    fetchCalls,
    historyCalls,
    scheduledTimeouts,
    setSelection: (txt) => { activeSelectionText = txt; },
    setContextHandler: (handler) => { contextFetchHandler = handler; },
    setStatusHandler: (handler) => { statusFetchHandler = handler; },
    modeInputs,
  };
}

// Suite 1: Title, explicit button, and non-interactive body activation without duplicate requests
{
  const env = await createTestEnv();
  const { vmContext, elements, fetchCalls, setSelection } = env;

  const mockResult = {
    conversation_id: "c-100",
    message_id: "m-200",
    source_id: "chatgpt",
    account_id: "test-user",
    title: "Quantum Computing Basics",
    text: "Quantum superposition allows qubits to exist in multiple states.",
    matches: [[8, 21]],
    match_type: "exact",
    match_count: 1,
  };

  vmContext.renderResults({ results: [mockResult], total: 1 });
  const resultsBox = elements.get("results");
  const card = resultsBox.querySelector(".result-card");
  assert(card, "Result card must be rendered");

  const titleBtn = card.querySelector(".result-title");
  assert(titleBtn, "Title button must exist");
  assert.equal(titleBtn.tag, "button", "Title must be a native button for keyboard access");
  assert.equal(titleBtn.attributes.get("type") || titleBtn.type, "button", "Title button must have type=button");

  const openBtn = card.querySelector(".open-conversation-btn");
  assert(openBtn, "Explicit Open conversation button must exist");
  assert.equal(openBtn.tag, "button", "Open action must be a native button for keyboard access");
  assert.equal(openBtn.attributes.get("type") || openBtn.type, "button", "Open button must have type=button");
  assert.equal(card.tag, "article", "Result card container must not be an interactive button role");

  // Title click: triggers single context fetch without duplicate
  const initialCount = fetchCalls.filter((c) => c.url.startsWith("/api/context")).length;
  titleBtn.click();
  await new Promise((r) => setImmediate(r));
  const afterTitle = fetchCalls.filter((c) => c.url.startsWith("/api/context")).length;
  assert.equal(afterTitle - initialCount, 1, "Title click must trigger exactly one context request");

  // Open conversation button click: triggers single context fetch without duplicate
  openBtn.click();
  await new Promise((r) => setImmediate(r));
  const afterOpenBtn = fetchCalls.filter((c) => c.url.startsWith("/api/context")).length;
  assert.equal(afterOpenBtn - afterTitle, 1, "Open button click must trigger exactly one context request");

  // Non-interactive card/excerpt click: triggers single context fetch
  const excerpt = card.querySelector(".result-excerpt");
  assert(excerpt, "Excerpt must exist");
  excerpt.click();
  await new Promise((r) => setImmediate(r));
  const afterExcerpt = fetchCalls.filter((c) => c.url.startsWith("/api/context")).length;
  assert.equal(afterExcerpt - afterOpenBtn, 1, "Non-interactive body click must open context");

  // Card click with active text selection: must NOT trigger context request, preserving selection
  setSelection("superposition allows");
  excerpt.click();
  await new Promise((r) => setImmediate(r));
  const afterSelectionClick = fetchCalls.filter((c) => c.url.startsWith("/api/context")).length;
  assert.equal(afterSelectionClick, afterExcerpt, "Clicking with active text selection must not open context");
  setSelection("");
}

// Suite 2: Context params namespace identity
{
  const env = await createTestEnv();
  const { vmContext, elements, fetchCalls } = env;

  const scopedResult = {
    conversation_id: "conv-custom-77",
    message_id: "msg-custom-88",
    source_id: "custom_source_archive",
    account_id: "user_account_99",
    title: "Scoped Context Check",
    text: "Testing preservation of source and account identity.",
    match_type: "exact",
  };

  vmContext.renderResults({ results: [scopedResult], total: 1 });
  const openBtn = elements.get("results").querySelector(".open-conversation-btn");
  openBtn.click();
  await new Promise((r) => setImmediate(r));

  const lastContextCall = fetchCalls.filter((c) => c.url.startsWith("/api/context")).pop();
  assert(lastContextCall, "A context request must have been made");
  const params = new URL("http://localhost" + lastContextCall.url).searchParams;
  assert.equal(params.get("conversation_id"), "conv-custom-77");
  assert.equal(params.get("message_id"), "msg-custom-88");
  assert.equal(params.get("source_id"), "custom_source_archive", "source_id namespace must be preserved");
  assert.equal(params.get("account_id"), "user_account_99", "account_id namespace must be preserved");
}

// Suite 3: Panel visibility and focus (so it cannot silently open below fold)
{
  const env = await createTestEnv();
  const { vmContext, elements } = env;

  const mockResult = {
    conversation_id: "c-vis",
    message_id: "m-vis",
    title: "Visibility Conversation",
    text: "Check visibility and focus behavior.",
  };

  vmContext.renderResults({ results: [mockResult], total: 1 });
  const placeholder = elements.get("context-placeholder");
  const content = elements.get("context-content");
  const pane = elements.get("context-pane");

  assert.equal(content.hidden, true, "Context content must be hidden initially");
  assert.equal(placeholder.hidden, false, "Placeholder must be visible initially");

  vmContext.openContext(mockResult, 0);

  assert.equal(placeholder.hidden, true, "Placeholder must be hidden when dialogue opens");
  assert.equal(content.hidden, false, "Content must be shown when dialogue opens");
  assert(pane.classList.contains("mobile-open"), "Panel must have mobile-open class");
  assert(pane.focused, "Dialogue panel must receive focus so it cannot open silently below the fold");
  assert(pane.scrolledIntoView, "Dialogue panel must be scrolled into view");

  const card = elements.get("results").querySelector(".result-card");
  assert(card.classList.contains("selected"), "Selected card must be visibly marked");
  assert.equal(card.getAttribute("aria-current"), "true", "Selected card must have aria-current=true");
}

// Suite 4: Semantic-only long Unicode message anchor with untouched source text
{
  const env = await createTestEnv();
  const { vmContext, elements, setContextHandler } = env;

  const prefix = "😀".repeat(500);
  const semanticPassage = "RELEVANT_SEMANTIC_FOCUS_PASSAGE";
  const suffix = "🌟".repeat(500);
  const originalAstralText = prefix + semanticPassage + suffix;
  const unicodeFocusOffset = 500;

  const semanticResult = {
    conversation_id: "c-astral",
    message_id: "m-astral",
    source_id: "chatgpt",
    account_id: "default",
    title: "Semantic Astral Dialogue",
    text: originalAstralText,
    matches: [],
    match_type: "semantic",
    focus_offset: unicodeFocusOffset,
  };

  setContextHandler(async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      title: "Semantic Astral Dialogue",
      url: "https://chatgpt.com/c/c-astral",
      messages: [
        { id: "m-astral", role: "assistant", text: originalAstralText },
      ],
    }),
  }));

  vmContext.renderResults({ results: [semanticResult], total: 1 });
  await vmContext.openContext(semanticResult, 0);
  await new Promise((r) => setImmediate(r));

  const messagesBox = elements.get("messages");
  const matchedSection = messagesBox.querySelector(".matched");
  assert(matchedSection, "Matched message section must have .matched class");

  const messageText = matchedSection.querySelector(".message-text");
  assert(messageText, "Message text container must exist");

  // Preserves untouched full original text:
  assert.equal(messageText.text, originalAstralText, "Original astral Unicode text must be strictly untouched");
  assert(!messageText.text.includes(String.fromCodePoint(0xFFFD)), "No split surrogate pairs or corrupted astral characters");

  // Semantic results must NOT mark similarity as exact phrase match:
  const marks = messageText.querySelectorAll("mark");
  assert.equal(marks.length, 0, "Semantic similarity must not be marked with exact phrase highlight marks");

  // Empty DOM anchor placed at validated Unicode code point offset:
  const anchor = messageText.querySelector(".semantic-anchor");
  assert(anchor, "Empty DOM anchor must be inserted for semantic-only focus");
  assert.equal(anchor.textContent, "", "Anchor must be an empty DOM element");
  assert.equal(anchor.getAttribute("aria-hidden"), "true", "Anchor must be aria-hidden");
  assert(anchor.scrolledIntoView, "Anchor must be scrolled into view for semantic passage focus");

  // Safe fallback for missing / invalid offsets:
  for (const invalidOffset of [null, undefined, "100", NaN, Infinity, -25, 15.5]) {
    const invalidResult = { ...semanticResult, focus_offset: invalidOffset };
    await vmContext.openContext(invalidResult, 0);
    await new Promise((r) => setImmediate(r));
    const fallbackText = messagesBox.querySelector(".matched").querySelector(".message-text");
    assert.equal(fallbackText.text, originalAstralText, "Invalid offset must safely fall back without text corruption");
    const fallbackAnchor = fallbackText.querySelector(".semantic-anchor");
    assert.equal(fallbackAnchor, null, "Invalid offset must safely fall back without inserting invalid anchor");
  }

  // Invalid beyond-end full-reader case: enormous offset must fall back to matched section, not jump to end
  const beyondEndResult = { ...semanticResult, focus_offset: 9999999 };
  await vmContext.openContext(beyondEndResult, 0);
  await new Promise((r) => setImmediate(r));
  const beyondMatched = messagesBox.querySelector(".matched");
  assert(beyondMatched, "Matched section must exist");
  const beyondText = beyondMatched.querySelector(".message-text");
  assert.equal(beyondText.text, originalAstralText, "Beyond-end offset must leave original text untouched");
  const beyondAnchor = beyondText.querySelector(".semantic-anchor");
  assert.equal(beyondAnchor, null, "Beyond-end offset must not insert anchor or jump to end");
  assert(beyondMatched.scrolledIntoView, "Beyond-end offset must safely fall back to scrolling matched section into view");
}

// Suite 5: Errors, close, and late responses
{
  const env = await createTestEnv();
  const { vmContext, elements, setContextHandler } = env;

  const mockResult = {
    conversation_id: "c-err",
    message_id: "m-err",
    title: "Error Handling Test",
    text: "Will fail on context fetch.",
  };

  setContextHandler(async () => {
    return {
      ok: false,
      status: 500,
      json: async () => ({ error: "Synthetic database read timeout" }),
    };
  });

  vmContext.renderResults({ results: [mockResult], total: 1 });
  await vmContext.openContext(mockResult, 0);
  await new Promise((r) => setImmediate(r));

  const card = elements.get("results").querySelector(".result-card");
  assert(card.classList.contains("error"), "Card must receive error class on failure");
  assert(!card.classList.contains("loading"), "Loading class must be removed on failure");

  const messagesBox = elements.get("messages");
  const errorPara = messagesBox.querySelector(".error-text");
  assert(errorPara, "Error text must be rendered in messages pane");
  assert(errorPara.textContent.includes("Synthetic database read timeout"));

  // Close context cleans up state
  vmContext.closeContext();
  assert.equal(elements.get("context-placeholder").hidden, false);
  assert.equal(elements.get("context-content").hidden, true);
  assert(!card.classList.contains("selected"), "Selected class must be removed on close");
  assert(!card.classList.contains("error"), "Error class must be removed on close");
  assert.equal(card.getAttribute("aria-current"), null);

  // Late response cancellation
  let resolveSlow = null;
  const slowPromise = new Promise((resolve) => { resolveSlow = resolve; });
  setContextHandler(async (url) => {
    if (url.includes("conv-slow")) {
      await slowPromise;
      return {
        ok: true, status: 200,
        json: async () => ({ title: "Slow Title - Should Be Discarded", messages: [] }),
      };
    }
    return {
      ok: true, status: 200,
      json: async () => ({ title: "Fast Title - Active", messages: [] }),
    };
  });

  const resSlow = { conversation_id: "conv-slow", message_id: "m-slow", title: "Slow" };
  const resFast = { conversation_id: "conv-fast", message_id: "m-fast", title: "Fast" };

  // Trigger slow request
  const p1 = vmContext.openContext(resSlow, 0);
  // Trigger fast request immediately after
  const p2 = vmContext.openContext(resFast, 0);
  await p2;
  assert.equal(elements.get("context-title").textContent, "Fast Title - Active");

  // Now resolve the slow request late
  resolveSlow();
  await p1;
  assert.equal(elements.get("context-title").textContent, "Fast Title - Active", "Late response must not overwrite newer request");
}

// Suite 6: Partial coverage and provider-error fallback / inert markup
{
  const env = await createTestEnv();
  const { vmContext, elements } = env;

  // Status readiness partial coverage against total messages:
  const partialSemantic = {
    configured: true,
    model_id: "sol-model",
    models: [{ model_id: "sol-model", indexed_messages: 25, vectors: 50 }],
  };
  const partialText = vmContext.semanticText(partialSemantic, 100);
  assert.equal(partialText, "25 of 100 messages ready for meaning search (partial)", "Partial coverage against total messages must be stated honestly");

  const fullText = vmContext.semanticText(partialSemantic, 25);
  assert.equal(fullText, "25 messages ready for meaning search", "Full coverage must not claim partial");

  // Per-query partial coverage and purpose description in smart mode:
  vm.runInContext("state.mode = 'smart';", vmContext);
  vmContext.renderResults({
    results: [{ title: "R1", text: "text 1", match_type: "semantic" }],
    total: 1,
    semantic: {
      used: true,
      indexed_messages: 120,
      pending_messages: 30,
    },
  });
  const modeNote = elements.get("mode-note").textContent;
  assert(modeNote.includes("related dialogue when the wording differs"), "Smart mode note must state honest dialogue purpose");
  assert(modeNote.includes("Search by a remembered idea or exchange"), "Smart mode note must state honest idea or exchange description");
  assert(modeNote.includes("Meaning coverage is partial: 120 indexed, 30 pending"), "Smart mode note must show per-query partial coverage");

  // Provider-error fallback: semantic.used is explicitly false while retaining purpose description
  vmContext.renderResults({
    results: [{ title: "R1", text: "text 1", match_type: "keyword" }],
    total: 1,
    semantic: {
      used: false,
      available: true, // available flag was true but used is false due to provider error
      error: "embedding provider rate limit",
    },
  });
  const fallbackNote = elements.get("mode-note").textContent;
  assert(fallbackNote.includes("Keyword fallback is active (embedding provider rate limit)"), "Keyword fallback must be labeled honestly");
  assert(/meaning search was not used/i.test(fallbackNote), "Must never claim meaning was used when semantic.used is false");
  assert(fallbackNote.includes("related dialogue when the wording differs"), "Keyword fallback note must retain honest dialogue purpose");

  // Actual backend contract: not_configured status
  vmContext.renderResults({
    results: [{ title: "R1", text: "text 1", match_type: "keyword" }],
    total: 1,
    semantic: {
      available: true,
      used: false,
      status: "not_configured",
    },
  });
  const notConfiguredNote = elements.get("mode-note").textContent;
  assert(notConfiguredNote.includes("Keyword fallback is active (meaning search not configured)"), "not_configured status must be reported honestly");
  assert(/meaning search was not used/i.test(notConfiguredNote), "Must never claim meaning was used when used is false");
  assert(notConfiguredNote.includes("related dialogue when the wording differs"), "Must retain honest dialogue purpose");

  // Actual backend contract: index_pending status with partial coverage
  vmContext.renderResults({
    results: [{ title: "R1", text: "text 1", match_type: "keyword" }],
    total: 1,
    semantic: {
      available: true,
      used: false,
      status: "index_pending",
      indexed_messages: 50,
      pending_messages: 150,
    },
  });
  const pendingNote = elements.get("mode-note").textContent;
  assert(pendingNote.includes("Keyword fallback is active (meaning index pending)"), "index_pending status must be reported honestly");
  assert(pendingNote.includes("Meaning coverage: 50 indexed, 150 pending (partial)"), "Partial coverage must be reported honestly on index pending");
  assert(/meaning search was not used/i.test(pendingNote), "Must never claim meaning was used when used is false");
  assert(pendingNote.includes("related dialogue when the wording differs"), "Must retain honest dialogue purpose");

  // Actual backend contract: provider_error with errors array, partial coverage, and inert markup
  vmContext.renderResults({
    results: [{ title: "R1", text: "text 1", match_type: "keyword" }],
    total: 1,
    semantic: {
      available: true,
      used: false,
      status: "provider_error",
      errors: ['<b onmouseover="alert(1)">rate limit exceeded</b>'],
      indexed_messages: 100,
      pending_messages: 25,
    },
  });
  const providerErrorNoteEl = elements.get("mode-note");
  assert.equal(providerErrorNoteEl.children.length, 0, "Error markup from errors[] must be rendered inertly without HTML execution");
  const providerErrorNote = providerErrorNoteEl.textContent;
  assert(providerErrorNote.includes("Keyword fallback is active (<b onmouseover=\"alert(1)\">rate limit exceeded</b>)"), "errors[0] must be reported in fallback note");
  assert(providerErrorNote.includes("Meaning coverage: 100 indexed, 25 pending (partial)"), "Partial coverage must be reported honestly on provider error");
  assert(/meaning search was not used/i.test(providerErrorNote), "Must never claim meaning was used when used is false");
  assert(providerErrorNote.includes("related dialogue when the wording differs"), "Must retain honest dialogue purpose");
  // Inert markup test in error note:
  vmContext.renderResults({
    results: [{ title: "R1", text: "text 1" }],
    total: 1,
    semantic: {
      used: false,
      error: '<img src=x onerror="alert(1)">',
    },
  });
  const inertNote = elements.get("mode-note");
  assert.equal(inertNote.children.length, 0, "Error markup must be rendered inertly without HTML execution");
  assert(inertNote.textContent.includes('<img src=x onerror="alert(1)">'));

  // Assert actual mode-note after real renderResults success and fallback includes honest purpose description:
  assert(modeNote.includes("related dialogue when the wording differs"), "Mode note on success must include honest purpose description");
  assert(fallbackNote.includes("related dialogue when the wording differs"), "Mode note on fallback must include honest purpose description");
}

// Suite 7: Smart pause when semantics unconfigured / unavailable
{
  // 7a: Unavailable config + mode=smart URL forcing exact, disabled controls and zero enrich calls
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: false, cloud_rerank_available: false },
      location: "http://localhost:8080/?mode=smart&q=wifi",
    });
    const { vmContext, elements, fetchCalls, historyCalls, modeInputs } = env;

    const [modeExact, modeSmart] = modeInputs;
    assert.equal(modeSmart.disabled, true, "Smart radio must be disabled when semantic indexing is unconfigured");
    assert.equal(modeSmart.checked, false, "Smart radio must be unchecked when semantic indexing is unconfigured");
    assert.equal(modeExact.checked, true, "Exact radio must be checked when semantic indexing is unconfigured");
    assert.equal(modeExact.disabled, false, "Exact radio must remain enabled");

    const indexBtn = elements.get("index-semantics");
    assert.equal(indexBtn.hidden, true, "Index button must be hidden when semantic is unavailable");
    assert.equal(indexBtn.disabled, true, "Index button must be disabled when semantic is unavailable");

    const modeNote = elements.get("mode-note").textContent;
    assert(/smart.*off/i.test(modeNote), "Mode note must clearly indicate Smart search is off");
    assert(/exact.*available/i.test(modeNote), "Mode note must clearly indicate Exact search is available");

    // Incoming mode=smart must be ignored and forced to exact in runSearch while unavailable
    const searchCalls = fetchCalls.filter((c) => c.url.startsWith("/api/search"));
    assert(searchCalls.length > 0, "Initial search must run for incoming query");
    const lastSearch = searchCalls[searchCalls.length - 1];
    const searchUrl = new URL("http://localhost" + lastSearch.url);
    assert.equal(searchUrl.searchParams.get("mode"), "exact", "Search request must force mode=exact when semantic is unavailable");
    assert.equal(searchUrl.searchParams.get("q"), "wifi");

    // History URL must reflect forced exact mode
    assert(historyCalls.length > 0, "History replaceState must be invoked");
    const lastHistory = historyCalls[historyCalls.length - 1];
    const historyUrl = new URL(lastHistory.url, "http://localhost:8080");
    assert.equal(historyUrl.searchParams.get("mode"), "exact", "History URL must force mode=exact");

    // Hidden index button handler must not request /api/enrich while unavailable
    const enrichCallsBefore = fetchCalls.filter((c) => c.url.startsWith("/api/enrich")).length;
    indexBtn.click();
    for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
    const enrichCallsAfter = fetchCalls.filter((c) => c.url.startsWith("/api/enrich")).length;
    assert.equal(enrichCallsAfter - enrichCallsBefore, 0, "Hidden/disabled index click must not request /api/enrich while unavailable");

    // Changing mode attempts while unavailable must force exact
    modeSmart.dispatchEvent({ type: "change" });
    for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
    assert.equal(modeSmart.checked, false, "Smart radio must stay unchecked");
    assert.equal(modeExact.checked, true, "Exact radio must stay checked");
  }

  // 7b: Configured path staying available
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: true, cloud_rerank_available: false },
      location: "http://localhost:8080/?mode=smart&q=wifi",
    });
    const { vmContext, elements, fetchCalls, modeInputs } = env;

    const [modeExact, modeSmart] = modeInputs;
    assert.equal(modeSmart.disabled, false, "Smart radio must be enabled when semantic is available");
    assert.equal(modeSmart.checked, true, "Smart radio must be checked when incoming mode=smart and semantic is available");
    assert.equal(modeExact.checked, false, "Exact radio must be unchecked when incoming mode=smart");

    const indexBtn = elements.get("index-semantics");
    assert.equal(indexBtn.hidden, false, "Index button must not be hidden when semantic is available");

    const searchCalls = fetchCalls.filter((c) => c.url.startsWith("/api/search"));
    assert(searchCalls.length > 0, "Initial search must run");
    const lastSearch = searchCalls[searchCalls.length - 1];
    const searchUrl = new URL("http://localhost" + lastSearch.url);
    assert.equal(searchUrl.searchParams.get("mode"), "smart", "Search request must preserve mode=smart when available");

    // Index button click requests /api/enrich when available
    const enrichCallsBefore = fetchCalls.filter((c) => c.url.startsWith("/api/enrich")).length;
    indexBtn.click();
    for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
    const enrichCallsAfter = fetchCalls.filter((c) => c.url.startsWith("/api/enrich")).length;
    assert.equal(enrichCallsAfter - enrichCallsBefore, 1, "Index click when available must call /api/enrich");
  }

  // 7c: Unavailable config + mode=smart without query forces exact and updates URL
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: false, cloud_rerank_available: false },
      location: "http://localhost:8080/?mode=smart",
    });
    const { elements, historyCalls, modeInputs } = env;
    const [modeExact, modeSmart] = modeInputs;
    assert.equal(modeSmart.disabled, true);
    assert.equal(modeSmart.checked, false);
    assert.equal(modeExact.checked, true);
    const modeNote = elements.get("mode-note").textContent;
    assert(/smart.*off/i.test(modeNote));
    assert(/exact.*available/i.test(modeNote));
    assert(historyCalls.length > 0);
    const lastHistory = historyCalls[historyCalls.length - 1];
    const historyUrl = new URL(lastHistory.url, "http://localhost:8080");
    assert.equal(historyUrl.searchParams.get("mode"), "exact");
  }

  // 7d: Configured path with running job at startup: index stays disabled and status polling remains scheduled
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: true, cloud_rerank_available: false },
      status: {
        conversations: 2,
        messages: 20,
        semantic_job: { state: "running" },
      },
      location: "http://localhost:8080/",
    });
    const { elements, fetchCalls, scheduledTimeouts } = env;

    const indexBtn = elements.get("index-semantics");
    assert.equal(indexBtn.hidden, false, "Index button must not be hidden when semantic is available");
    assert.equal(indexBtn.disabled, true, "Index button must remain disabled while background job is running");
    assert.equal(indexBtn.textContent, "Indexing meaning…", "Button text must show indexing in progress");

    // Existing status polling must remain scheduled without any enrich/model call
    assert(scheduledTimeouts.some((t) => t.delay === 1800), "Status polling timer must remain scheduled when job is running");
    const enrichCalls = fetchCalls.filter((c) => c.url.startsWith("/api/enrich")).length;
    assert.equal(enrichCalls, 0, "No enrich calls must be made during running-job startup");
  }
}

// Suite 8: Stale dialogue invalidation and race conditions
{
  // 8a: Deferred old context resolving after new result list (renderResults replacement)
  {
    const env = await createTestEnv();
    const { vmContext, elements, setContextHandler } = env;

    let resolveOldContext;
    const oldContextPromise = new Promise((resolve) => { resolveOldContext = resolve; });

    setContextHandler(async (url) => {
      if (url.includes("conv-old")) {
        await oldContextPromise;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            title: "Old Stale Conversation",
            messages: [{ id: "m-old", role: "user", text: "Stale conversation content" }],
          }),
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          title: "New Conversation",
          messages: [{ id: "m-new", role: "assistant", text: "Fresh conversation content" }],
        }),
      };
    });

    const oldResult = { conversation_id: "conv-old", message_id: "m-old", title: "Old Title", text: "old text" };
    vmContext.renderResults({ results: [oldResult], total: 1 });

    const openPromise = vmContext.openContext(oldResult, 0);

    const placeholder = elements.get("context-placeholder");
    const content = elements.get("context-content");
    const messages = elements.get("messages");

    assert.equal(placeholder.hidden, true);
    assert.equal(content.hidden, false);
    assert.equal(elements.get("context-title").textContent, "Old Title");
    assert(messages.querySelector(".loading"));

    // Result set is replaced with new results
    const newResult = { conversation_id: "conv-new", message_id: "m-new", title: "New Title", text: "new text" };
    vmContext.renderResults({ results: [newResult], total: 1 });

    // Result replacement must invalidate prior context viewer & selected state
    assert.equal(placeholder.hidden, false, "Placeholder must be visible after result set replacement");
    assert.equal(content.hidden, true, "Context content must be hidden after result set replacement");
    assert.equal(vm.runInContext("state.selected", vmContext), null, "Selected state must be cleared after result set replacement");

    const newCards = elements.get("results").querySelectorAll(".result-card");
    assert.equal(newCards.length, 1);
    assert(!newCards[0].classList.contains("selected"), "New card must not inherit selected state");
    assert(!newCards[0].classList.contains("loading"), "New card must not inherit loading state");

    // Deferred old context resolves now
    resolveOldContext();
    await openPromise;

    // Must not reopen, paint, or corrupt viewer
    assert.equal(placeholder.hidden, false, "Placeholder must stay visible when stale old context resolves");
    assert.equal(content.hidden, true, "Context content must stay hidden when stale old context resolves");
    assert.equal(vm.runInContext("state.selected", vmContext), null, "Selected state must stay null");
    assert(!newCards[0].classList.contains("selected"), "New card must not become selected");
    assert(!newCards[0].classList.contains("loading"), "New card must not become loading");
    assert(!messages.text.includes("Stale conversation content"), "Stale old messages must not be painted");

    // Subsequent new result opens cleanly
    await vmContext.openContext(newResult, 0);
    assert.equal(placeholder.hidden, true);
    assert.equal(content.hidden, false);
    assert.equal(elements.get("context-title").textContent, "New Conversation");
    assert(messages.text.includes("Fresh conversation content"), "Fresh conversation content must be displayed");
    assert(newCards[0].classList.contains("selected"));
    assert(!newCards[0].classList.contains("loading"));
  }

  // 8b: Deferred old context resolving after query cleared
  {
    const env = await createTestEnv();
    const { vmContext, elements, setContextHandler } = env;

    let resolveOldContext;
    const oldContextPromise = new Promise((resolve) => { resolveOldContext = resolve; });

    setContextHandler(async (url) => {
      if (url.includes("conv-query-old")) {
        await oldContextPromise;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            title: "Stale Query Conversation",
            messages: [{ id: "m-q-old", role: "user", text: "Stale query text" }],
          }),
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          title: "Subsequent Title",
          messages: [{ id: "m-sub", role: "assistant", text: "Subsequent text" }],
        }),
      };
    });

    const oldResult = { conversation_id: "conv-query-old", message_id: "m-q-old", title: "Query Old Title", text: "query old" };
    vmContext.renderResults({ results: [oldResult], total: 1 });
    const openPromise = vmContext.openContext(oldResult, 0);

    const queryInput = elements.get("query");
    queryInput.value = "";
    queryInput.focus();

    // Run search with cleared query (e.g. typing backspace)
    await vmContext.runSearch(true);

    const placeholder = elements.get("context-placeholder");
    const content = elements.get("context-content");

    assert.equal(placeholder.hidden, false, "Placeholder must be visible when query is cleared");
    assert.equal(content.hidden, true, "Context content must be hidden when query is cleared");
    assert.equal(vm.runInContext("state.selected", vmContext), null);
    assert.equal(queryInput.focused, true, "Focus must not be stolen from query input on search changes");

    // Old context completes late
    resolveOldContext();
    await openPromise;

    assert.equal(placeholder.hidden, false, "Stale context resolution must not reopen viewer when query was cleared");
    assert.equal(content.hidden, true, "Context content must stay hidden");
    assert.equal(vm.runInContext("state.selected", vmContext), null);

    // Subsequent search and result opening works correctly
    const subsequentResult = { conversation_id: "conv-sub", message_id: "m-sub", title: "Subsequent Title", text: "sub" };
    vmContext.renderResults({ results: [subsequentResult], total: 1 });
    const cards = elements.get("results").querySelectorAll(".result-card");
    assert.equal(cards.length, 1);

    await vmContext.openContext(subsequentResult, 0);
    assert.equal(placeholder.hidden, true);
    assert.equal(content.hidden, false);
    assert.equal(elements.get("context-title").textContent, "Subsequent Title");
    assert(elements.get("messages").text.includes("Subsequent text"));
    assert(cards[0].classList.contains("selected"));
  }

  // 8c: Deferred old context resolving after new search starts
  {
    const env = await createTestEnv();
    const { vmContext, elements, setContextHandler } = env;

    let resolveOldContext;
    const oldContextPromise = new Promise((resolve) => { resolveOldContext = resolve; });

    setContextHandler(async (url) => {
      if (url.includes("conv-search-old")) {
        await oldContextPromise;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            title: "Stale Search Conv",
            messages: [{ id: "m-s-old", role: "user", text: "Stale search message" }],
          }),
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          title: "Subsequent Search Conv",
          messages: [{ id: "m-s-new", role: "assistant", text: "New search message" }],
        }),
      };
    });

    const oldResult = { conversation_id: "conv-search-old", message_id: "m-s-old", title: "Search Old Title", text: "old" };
    vmContext.renderResults({ results: [oldResult], total: 1 });
    const openPromise = vmContext.openContext(oldResult, 0);

    const queryInput = elements.get("query");
    queryInput.value = "quantum";
    queryInput.focus();

    // Start new search
    const searchPromise = vmContext.runSearch(true);

    const placeholder = elements.get("context-placeholder");
    const content = elements.get("context-content");

    assert.equal(placeholder.hidden, false, "Placeholder must be visible when new search starts");
    assert.equal(content.hidden, true, "Context content must be hidden when new search starts");
    assert.equal(vm.runInContext("state.selected", vmContext), null);
    assert.equal(queryInput.focused, true, "Focus must not be stolen from query input on new search");

    // Old context finishes late
    resolveOldContext();
    await openPromise;
    await searchPromise;

    assert.equal(placeholder.hidden, false, "Stale context must not reopen viewer");
    assert.equal(content.hidden, true);
    assert.equal(vm.runInContext("state.selected", vmContext), null);

    // Subsequent result opening works properly
    const newResult = { conversation_id: "conv-search-new", message_id: "m-s-new", title: "Subsequent Search Conv", text: "new" };
    vmContext.renderResults({ results: [newResult], total: 1 });
    await vmContext.openContext(newResult, 0);

    assert.equal(placeholder.hidden, true);
    assert.equal(content.hidden, false);
    assert.equal(elements.get("context-title").textContent, "Subsequent Search Conv");
    assert(elements.get("messages").text.includes("New search message"));
  }
}

// Suite 9: Exact match focus on first verified literal mark inside long messages
{
  const env = await createTestEnv();
  const { vmContext, elements, setContextHandler } = env;

  // 9a: Deep exact hit after astral Unicode characters
  const astralPrefix = "🚀".repeat(300) + "✨".repeat(200); // 500 code points
  const matchTerm1 = "VERIFIED_LITERAL_KEYWORD";
  const intermediateAstral = "🌍".repeat(200);
  const matchTerm2 = "SECOND_LITERAL_OCCURRENCE";
  const astralSuffix = "🎉".repeat(300);
  const fullAstralText = astralPrefix + matchTerm1 + intermediateAstral + matchTerm2 + astralSuffix;

  // Compute exact code point ranges
  const prefixLength = Array.from(astralPrefix).length; // 500
  const term1Length = Array.from(matchTerm1).length;
  const start1 = prefixLength;
  const end1 = prefixLength + term1Length;

  const intermediateLength = Array.from(intermediateAstral).length;
  const start2 = end1 + intermediateLength;
  const end2 = start2 + Array.from(matchTerm2).length;

  const exactAstralResult = {
    conversation_id: "c-exact-astral",
    message_id: "m-exact-astral",
    source_id: "chatgpt",
    account_id: "default",
    title: "Deep Exact Astral Dialogue",
    text: fullAstralText,
    matches: [[start1, end1], [start2, end2]],
    match_type: "exact",
    match_count: 2,
    focus_offset: start1,
  };

  setContextHandler(async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      title: "Deep Exact Astral Dialogue",
      url: "https://chatgpt.com/c/c-exact-astral",
      messages: [
        { id: "m-exact-astral", role: "assistant", text: fullAstralText },
      ],
    }),
  }));

  vmContext.renderResults({ results: [exactAstralResult], total: 1 });
  await vmContext.openContext(exactAstralResult, 0);
  await new Promise((r) => setImmediate(r));

  const messagesBox = elements.get("messages");
  const matchedSection = messagesBox.querySelector(".matched");
  assert(matchedSection, "Matched section must exist");

  const messageText = matchedSection.querySelector(".message-text");
  assert.equal(messageText.text, fullAstralText, "Original astral Unicode text must be strictly unchanged");

  const marks = messageText.querySelectorAll("mark");
  assert.equal(marks.length, 2, "Both verified literal occurrences must be rendered as marks");
  assert.equal(marks[0].textContent, matchTerm1, "First mark text must match exactly");
  assert.equal(marks[1].textContent, matchTerm2, "Second mark text must match exactly");

  // First verified literal mark must be scrolled into view
  assert.equal(marks[0].scrolledIntoView, true, "First verified literal mark must be scrolled into view");
  assert.equal(marks[0].scrollOptions?.block, "center", "First mark must scroll with block: center");

  // Second mark must not be scrolled into view
  assert.equal(marks[1].scrolledIntoView, false, "Second mark must not be scrolled into view");

  // No invented semantic anchor when literal marks exist
  const anchor = messageText.querySelector(".semantic-anchor");
  assert.equal(anchor, null, "No semantic-anchor element must be created when literal marks exist");

  // 9b: Multiple literal hits focusing the first verified mark even if out of order or overlapping
  {
    const textWithMultiple = "Intro AAA Middle BBB Outro";
    const multiResult = {
      conversation_id: "c-multi",
      message_id: "m-multi",
      title: "Multiple Hits",
      text: textWithMultiple,
      // Unsorted ranges: second occurrence listed before first
      matches: [[17, 20], [6, 9]],
      match_type: "exact",
      match_count: 2,
    };

    setContextHandler(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        title: "Multiple Hits",
        messages: [{ id: "m-multi", role: "assistant", text: textWithMultiple }],
      }),
    }));

    vmContext.renderResults({ results: [multiResult], total: 1 });
    await vmContext.openContext(multiResult, 0);
    await new Promise((r) => setImmediate(r));

    const curMatched = elements.get("messages").querySelector(".matched");
    const curMarks = curMatched.querySelectorAll("mark");
    assert.equal(curMarks.length, 2, "Both marks must be rendered");
    assert.equal(curMarks[0].textContent, "AAA", "First mark in document order must be AAA");
    assert.equal(curMarks[1].textContent, "BBB", "Second mark in document order must be BBB");
    assert.equal(curMarks[0].scrolledIntoView, true, "First mark must be scrolled into view");
    assert.equal(curMarks[1].scrolledIntoView, false, "Second mark must not be scrolled into view");
    assert.equal(curMatched.querySelector(".message-text").text, textWithMultiple, "Original text must be unchanged");
  }

  // 9c: Invalid and missing match spans fall back safely without text corruption or invented marks
  {
    const fallbackText = "Long message without valid marks to test fallback behavior.";
    for (const invalidMatches of [
      [],
      null,
      undefined,
      [[null, null]],
      [[-10, 5]],
      [[10, 5]], // start >= end
      [[5, 99999]], // exceeds length
      [["6", "9"]], // non-integer
      [[NaN, 10]],
    ]) {
      const invalidResult = {
        conversation_id: "c-fallback",
        message_id: "m-fallback",
        title: "Invalid Matches Fallback",
        text: fallbackText,
        matches: invalidMatches,
        match_type: "exact",
      };

      setContextHandler(async () => ({
        ok: true,
        status: 200,
        json: async () => ({
          title: "Invalid Matches Fallback",
          messages: [{ id: "m-fallback", role: "assistant", text: fallbackText }],
        }),
      }));

      vmContext.renderResults({ results: [invalidResult], total: 1 });
      await vmContext.openContext(invalidResult, 0);
      await new Promise((r) => setImmediate(r));

      const curMatched = elements.get("messages").querySelector(".matched");
      assert(curMatched, "Matched section must exist");
      const curText = curMatched.querySelector(".message-text");
      assert.equal(curText.text, fallbackText, "Original text must be completely intact");
      const curMarks = curText.querySelectorAll("mark");
      assert.equal(curMarks.length, 0, "No marks should be created for invalid match ranges");

      // Fallback: scrolls matched section into view
      assert.equal(curMatched.scrolledIntoView, true, "Matched section must be scrolled into view when no valid marks exist");
      assert.equal(curMatched.scrollOptions?.block, "start");
    }
  }

  // 9d: Fallback to semantic anchor when matches is empty/missing but valid focus_offset is present
  {
    const semanticFallbackText = "Hello " + "🌸".repeat(100) + " target idea in long text.";
    const semanticAnchorResult = {
      conversation_id: "c-sem-fallback",
      message_id: "m-sem-fallback",
      title: "Semantic Fallback With Anchor",
      text: semanticFallbackText,
      matches: [],
      match_type: "exact",
      focus_offset: 106,
    };

    setContextHandler(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        title: "Semantic Fallback With Anchor",
        messages: [{ id: "m-sem-fallback", role: "assistant", text: semanticFallbackText }],
      }),
    }));

    vmContext.renderResults({ results: [semanticAnchorResult], total: 1 });
    await vmContext.openContext(semanticAnchorResult, 0);
    await new Promise((r) => setImmediate(r));

    const curMatched = elements.get("messages").querySelector(".matched");
    const curText = curMatched.querySelector(".message-text");
    assert.equal(curText.text, semanticFallbackText, "Original text must be completely intact");
    const curMarks = curText.querySelectorAll("mark");
    assert.equal(curMarks.length, 0, "No marks created");

    const curAnchor = curText.querySelector(".semantic-anchor");
    assert(curAnchor, "Semantic anchor should be present when no literal marks and valid focus_offset");
    assert.equal(curAnchor.scrolledIntoView, true, "Semantic anchor must be scrolled into view");
    assert.equal(curAnchor.scrollOptions?.block, "center");
  }
}

// Suite 10: Explicit submit dialogue recall interface, input race cancellation, and dialogue preview
{
  // 10a: Smart input event does not debounce or schedule search; resets pending debounce and context viewer
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: true, cloud_rerank_available: false },
    });
    const { vmContext, elements, fetchCalls, scheduledTimeouts, modeInputs } = env;
    const [, modeSmart] = modeInputs;

    // Switch to smart
    modeInputs[0].checked = false;
    modeInputs[1].checked = true;
    modeSmart.dispatchEvent({ type: "change" });

    // Cue shown on entering smart mode
    const emptySymbol = elements.get("results").querySelector(".empty-state")?.querySelector("h3");
    assert(emptySymbol && emptySymbol.textContent.includes("Press Search or Enter to retrieve"), "Entering Smart mode must display concise cue waiting for explicit submit");

    const searchCallsBefore = fetchCalls.filter((c) => c.url.startsWith("/api/search")).length;
    const timeoutsBefore = scheduledTimeouts.length;

    // Type in query input
    const queryInput = elements.get("query");
    queryInput.value = "machine learning dialogue";
    queryInput.dispatchEvent({ type: "input" });

    // Must NOT schedule runSearch or fetch
    const searchCallsAfter = fetchCalls.filter((c) => c.url.startsWith("/api/search")).length;
    assert.equal(searchCallsAfter, searchCallsBefore, "Input event in Smart must not initiate fetch");

    const newTimeouts = scheduledTimeouts.slice(timeoutsBefore);
    assert.equal(newTimeouts.length, 0, "Input event in Smart must not schedule debounce timer");

    // Clearing input in Smart mode resets UI without calling API
    queryInput.value = "";
    queryInput.dispatchEvent({ type: "input" });
    const searchCallsAfterClear = fetchCalls.filter((c) => c.url.startsWith("/api/search")).length;
    assert.equal(searchCallsAfterClear, searchCallsBefore, "Clearing query in Smart must reset UI without API call");
    assert.equal(elements.get("results-title").textContent, "Describe a remembered exchange");
  }

  // 10b: Deferred race: late pending search/context cannot paint obsolete content when typing in Smart mode;
  // typing a different nonempty query leaves no old clickable result/pagination and clears busy
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: true, cloud_rerank_available: false },
    });
    const { vmContext, elements, setContextHandler } = env;

    let resolveSlowContext;
    const slowContextPromise = new Promise((resolve) => { resolveSlowContext = resolve; });

    setContextHandler(async (url) => {
      if (url.includes("conv-late-smart")) {
        await slowContextPromise;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            title: "Late Obsolete Context",
            messages: [{ id: "m-late", role: "assistant", text: "Obsolete text that must not paint" }],
          }),
        };
      }
      return { ok: true, status: 200, json: async () => ({}) };
    });

    const mockResult = {
      conversation_id: "conv-late-smart",
      message_id: "m-late",
      title: "Late Result",
      text: "late text",
    };
    vmContext.renderResults({ results: [mockResult], total: 100 });
    const openPromise = vmContext.openContext(mockResult, 0);

    assert.equal(elements.get("context-placeholder").hidden, true);
    assert.equal(elements.get("context-content").hidden, false);
    assert.equal(elements.get("results").querySelectorAll(".result-card").length, 1);
    assert.equal(elements.get("pagination").hidden, false);

    // Switch to Smart
    const [, modeSmart] = env.modeInputs;
    env.modeInputs[0].checked = false;
    env.modeInputs[1].checked = true;
    modeSmart.dispatchEvent({ type: "change" });

    // Entering Smart clears state results/total/offset, resets viewer and shows cue
    assert.equal(vm.runInContext("state.results.length", vmContext), 0);
    assert.equal(vm.runInContext("state.total", vmContext), 0);
    assert.equal(vm.runInContext("state.offset", vmContext), 0);
    assert.equal(elements.get("results").querySelectorAll(".result-card").length, 0);
    assert.equal(elements.get("pagination").hidden, true);
    assert.equal(elements.get("results").getAttribute("aria-busy"), "false");

    // Re-render old result to simulate prior active results before typing
    vmContext.renderResults({ results: [mockResult], total: 100 });
    assert.equal(elements.get("results").querySelectorAll(".result-card").length, 1);
    assert.equal(elements.get("pagination").hidden, false);

    // Type a different nonempty query in Smart:
    // leaves no old clickable result/pagination, clears busy, shows Press Search or Enter cue, sets query/offset0
    const queryInput = elements.get("query");
    queryInput.value = "different query";
    queryInput.dispatchEvent({ type: "input" });

    assert.equal(elements.get("results").querySelectorAll(".result-card").length, 0, "No old result cards must remain clickable on typing in Smart");
    assert.equal(elements.get("pagination").hidden, true, "Pagination must be hidden on typing in Smart");
    assert.equal(elements.get("results").getAttribute("aria-busy"), "false", "aria-busy must be false on typing in Smart");
    assert.equal(elements.get("results-title").textContent, "Describe a remembered exchange");
    assert.equal(elements.get("result-count").textContent, "");
    assert.equal(vm.runInContext("state.results.length", vmContext), 0);
    assert.equal(vm.runInContext("state.total", vmContext), 0);
    assert.equal(vm.runInContext("state.offset", vmContext), 0);
    assert.equal(vm.runInContext("state.query", vmContext), "different query");
    assert.equal(elements.get("context-placeholder").hidden, false, "Placeholder must be visible immediately on typing in Smart");
    assert.equal(elements.get("context-content").hidden, true, "Content must be hidden immediately on typing in Smart");

    const cue = elements.get("results").querySelector(".empty-state")?.querySelector("h3");
    assert(cue && cue.textContent.includes("Press Search or Enter to retrieve"), "Shows Press Search or Enter cue");

    // Late context finishes
    resolveSlowContext();
    await openPromise;

    // Must not paint obsolete content
    assert.equal(elements.get("context-placeholder").hidden, false, "Late context must not reopen viewer");
    assert.equal(elements.get("context-content").hidden, true);
    assert(!elements.get("messages").text.includes("Obsolete text that must not paint"), "Obsolete content must not be painted");
    assert.equal(elements.get("results").querySelectorAll(".result-card").length, 0, "No obsolete results painted");
  }

  // 10c: Successful explicit submit in Smart sends case_sensitive=0 even if checkbox was checked
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: true, cloud_rerank_available: false },
    });
    const { elements, fetchCalls, modeInputs } = env;
    const [, modeSmart] = modeInputs;

    elements.get("case-sensitive").checked = true;

    modeInputs[0].checked = false;
    modeInputs[1].checked = true;
    modeSmart.dispatchEvent({ type: "change" });

    assert.equal(elements.get("case-sensitive").disabled, true, "Case sensitive checkbox must be disabled in Smart mode");
    assert.equal(elements.get("query").placeholder, "Describe an idea or remembered dialogue…", "Mode-appropriate placeholder in Smart");

    elements.get("query").value = "distributed consensus";
    // Submit search form explicitly
    elements.get("search-form").dispatchEvent({ type: "submit", preventDefault() {} });
    await new Promise((r) => setImmediate(r));

    const smartSearchCalls = fetchCalls.filter((c) => c.url.startsWith("/api/search"));
    assert(smartSearchCalls.length > 0, "Explicit submit must execute search");
    const lastCall = smartSearchCalls[smartSearchCalls.length - 1];
    const params = new URL("http://localhost" + lastCall.url).searchParams;
    assert.equal(params.get("q"), "distributed consensus");
    assert.equal(params.get("mode"), "smart");
    assert.equal(params.get("case_sensitive"), "0", "Smart search must send case_sensitive=0 despite checkbox having been checked");
  }

  // 10d: Exact mode retains debouncing and checkbox value
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: true, cloud_rerank_available: false },
    });
    const { elements, fetchCalls, scheduledTimeouts, modeInputs } = env;
    const [modeExact] = modeInputs;

    modeInputs[1].checked = false;
    modeInputs[0].checked = true;
    modeExact.dispatchEvent({ type: "change" });
    elements.get("case-sensitive").checked = true;

    assert.equal(elements.get("query").placeholder, "Search exact words remembered…");

    const initialFetchCount = fetchCalls.filter((c) => c.url.startsWith("/api/search")).length;
    elements.get("query").value = "exact literal text";
    elements.get("query").dispatchEvent({ type: "input" });

    // Exact schedules debounce timer
    const debounceTimer = scheduledTimeouts.find((t) => t.delay === 250);
    assert(debounceTimer, "Exact search must debounce input by 250ms");

    // Execute debounced search
    await debounceTimer.fn();
    await new Promise((r) => setImmediate(r));

    const exactSearchCalls = fetchCalls.filter((c) => c.url.startsWith("/api/search"));
    assert.equal(exactSearchCalls.length, initialFetchCount + 1, "Debounced exact search must fetch");
    const lastCall = exactSearchCalls[exactSearchCalls.length - 1];
    const params = new URL("http://localhost" + lastCall.url).searchParams;
    assert.equal(params.get("case_sensitive"), "1", "Exact search must retain case_sensitive checkbox value");
  }

  // 10e: Smart per-card retrieval reason using known channels/match_type only (no scores/confidence)
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: true, cloud_rerank_available: false },
    });
    const { vmContext, elements, modeInputs } = env;
    const [, modeSmart] = modeInputs;
    modeInputs[0].checked = false;
    modeInputs[1].checked = true;
    modeSmart.dispatchEvent({ type: "change" });

    const testResults = [
      { conversation_id: "c1", message_id: "m1", title: "T1", text: "text1", match_type: "exact" },
      { conversation_id: "c2", message_id: "m2", title: "T2", text: "text2", match_type: "lexical" },
      { conversation_id: "c3", message_id: "m3", title: "T3", text: "text3", match_type: "semantic" },
      { conversation_id: "c4", message_id: "m4", title: "T4", text: "text4", match_type: "hybrid", retrieval_channels: ["literal", "keyword", "dialogue"] },
      { conversation_id: "c5", message_id: "m5", title: "T5", text: "text5", match_type: "lexical", retrieval_channels: ["keyword"] },
      { conversation_id: "c6", message_id: "m6", title: "T6", text: "text6", match_type: "other_unknown", retrieval_channels: ["unknown_chan"] },
      { conversation_id: "c7", message_id: "m7", title: "T7", text: "text7", match_type: "other", retrieval_channels: ["literal", "dialogue"] },
    ];

    vmContext.renderResults({ results: testResults, total: 7 });
    const cards = elements.get("results").querySelectorAll(".result-card");
    assert.equal(cards.length, 7);

    const reasons = Array.from(cards).map((c) => c.querySelector(".match-info").textContent);
    assert.equal(reasons[0], "exact phrase");
    assert.equal(reasons[1], "keyword match");
    assert.equal(reasons[2], "related dialogue");
    assert.equal(reasons[3], "words + meaning");
    assert.equal(reasons[4], "keyword match");
    assert.equal(reasons[5], "ranked passage");
    assert.equal(reasons[6], "words + meaning");

    // Ensure no numeric scores or confidence anywhere in cards
    const resultsContainerText = elements.get("results").text;
    assert(!/\b0\.\d+\b/.test(resultsContainerText), "No numeric confidence or scores displayed in result cards");
  }

  // 10f: Dialogue preview: up to 2 ancestor original excerpts, inert text, capped 240 chars, generic reply with prior question shown
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: true, cloud_rerank_available: false },
    });
    const { vmContext, elements, fetchCalls, setContextHandler, modeInputs } = env;
    const [, modeSmart] = modeInputs;
    modeInputs[0].checked = false;
    modeInputs[1].checked = true;
    modeSmart.dispatchEvent({ type: "change" });

    const genericYesResult = {
      conversation_id: "conv-preview-yes",
      message_id: "msg-yes-123",
      source_id: "chatgpt",
      account_id: "work-acct",
      title: "Confirming deployment settings",
      text: "Yes, exactly.",
      match_type: "semantic",
      dialogue_preview: [
        {
          message_id: "msg-prior-q",
          role: "user",
          text: '<img src=x onerror="alert(1)"> Does this architecture handle multi-region failover seamlessly?',
          start_offset: 0,
          end_offset: 75,
        },
        {
          message_id: "msg-prior-extra",
          role: "assistant",
          text: "Z".repeat(300), // Exceeds 240 chars
        },
        {
          message_id: "msg-third-ignored",
          role: "user",
          text: "Should be capped at 2 and not appear",
        },
      ],
    };

    vmContext.renderResults({ results: [genericYesResult], total: 1 });
    const card = elements.get("results").querySelector(".result-card");
    assert(card, "Card must be rendered");

    // Earlier in this exchange section
    const earlierBox = card.querySelector(".earlier-exchanges");
    assert(earlierBox, "Earlier in this exchange block must be rendered");
    const heading = earlierBox.querySelector(".earlier-heading");
    assert.equal(heading.textContent, "Earlier in this exchange");

    const items = earlierBox.querySelectorAll(".earlier-item");
    assert.equal(items.length, 2, "Previews must be capped at 2");

    // First item: generic yes reply has prior question shown with inert markup
    assert(items[0].text.includes("You"), "Role label must be displayed");
    assert(items[0].text.includes('<img src=x onerror="alert(1)">'), "Markup must be rendered inertly as text");
    assert.equal(items[0].querySelectorAll("img").length, 0, "No HTML tags executed in preview");

    // Second item: capped at 240 code points
    const item2Text = items[1].text;
    assert(item2Text.includes("AI"), "AI role label");
    // Text after role label should be capped at 240
    const rawZ = item2Text.replace(/^AI/, "");
    assert.equal(Array.from(rawZ).length, 240, "Preview text must be capped at 240 code points");

    // No preview fabricated literal marks in excerpt
    const excerpt = card.querySelector(".result-excerpt");
    assert.equal(excerpt.querySelectorAll("mark").length, 0, "No fabricated literal marks in semantic excerpt");

    // Context click on generic reply preserves original primary message_id, source_id, account_id
    const openBtn = card.querySelector(".open-conversation-btn");
    openBtn.click();
    await new Promise((r) => setImmediate(r));

    const lastContextCall = fetchCalls.filter((c) => c.url.startsWith("/api/context")).pop();
    assert(lastContextCall, "Context call must be made");
    const params = new URL("http://localhost" + lastContextCall.url).searchParams;
    assert.equal(params.get("conversation_id"), "conv-preview-yes");
    assert.equal(params.get("message_id"), "msg-yes-123", "Primary matched message_id must be requested, not preview message_id");
    assert.equal(params.get("source_id"), "chatgpt");
    assert.equal(params.get("account_id"), "work-acct");
  }

  // 10g: Malformed / absent dialogue_preview handled safely
  {
    const env = await createTestEnv({
      config: { token: "synthetic-token", semantic_available: true, cloud_rerank_available: false },
    });
    const { vmContext, elements, modeInputs } = env;
    const [, modeSmart] = modeInputs;
    modeInputs[0].checked = false;
    modeInputs[1].checked = true;
    modeSmart.dispatchEvent({ type: "change" });

    for (const badPreview of [null, undefined, "not an array", [null, "string item", 123]]) {
      vmContext.renderResults({
        results: [{
          conversation_id: "c-bad",
          message_id: "m-bad",
          title: "Bad Preview",
          text: "Safe text",
          dialogue_preview: badPreview,
        }],
        total: 1,
      });
      const card = elements.get("results").querySelector(".result-card");
      assert(card, "Card must render without throwing on malformed dialogue_preview");
      assert.equal(card.querySelector(".earlier-item")?.textContent || "", "");
    }
  }

  // 11: Explicit bounded batches, coverage controls, safe pause/resume, and retry remaining
  {
    const env = await createTestEnv({
      config: {
        token: "synthetic-token",
        semantic_available: true,
        cloud_rerank_available: false,
        enrich_controls_version: 1,
      },
      status: {
        conversations: 5,
        messages: 20,
        sources: [{ source_id: "chatgpt", account_id: "work" }],
        semantic: {
          configured: true,
          model_id: "test-model-v2",
          coverage: { account_id: null, eligible: 20, current: 0, pending: 20, legacy: 0 },
        },
        semantic_job: {
          job_id: "job-1234",
          state: "running",
          scope: { account_id: "work", limit: 100, generate_topics: false },
        },
      },
    });
    const { elements, vmContext, fetchCalls } = env;
    const indexBtn = elements.get("index-semantics");
    const pauseBtn = elements.get("pause-enrichment");
    const statusSpan = elements.get("semantic-status");

    // When running with enrich_controls_version >= 1:
    // pause button is visible and active; index button is disabled
    assert.equal(pauseBtn.hidden, false, "Pause button must be visible when running");
    assert.equal(pauseBtn.disabled, false, "Pause button must be enabled when running");
    assert.equal(pauseBtn.textContent, "Pause");
    assert.equal(pauseBtn.getAttribute("aria-label"), "Pause meaning indexing for work");
    assert.equal(indexBtn.disabled, true, "Index button must be disabled when running");
    assert.equal(statusSpan.textContent, "Background meaning indexing in progress (work)");

    // Clicking pause sends POST /api/enrich/pause with job_id and updates text
    pauseBtn.click();
    await new Promise((r) => setImmediate(r));
    const pauseCalls = fetchCalls.filter((c) => c.url === "/api/enrich/pause");
    assert.equal(pauseCalls.length, 1, "Clicking pause must post to /api/enrich/pause");
    assert.equal(JSON.parse(pauseCalls[0].options.body).job_id, "job-1234");
    assert.equal(pauseBtn.textContent, "Stopping after current request…");
    assert.equal(pauseBtn.disabled, true);

    // Paused state: pause button hidden, index button shows Resume
    elements.get("account").value = "work";
    env.setStatusHandler(() => ({
      conversations: 5,
      messages: 20,
      sources: [{ source_id: "chatgpt", account_id: "work" }, { source_id: "chatgpt", account_id: "other" }],
      semantic: {
        configured: true,
        model_id: "test-model-v2",
        coverage: { account_id: "work", eligible: 20, current: 10, pending: 10, legacy: 0 },
      },
      semantic_job: {
        job_id: "job-1234",
        state: "paused",
        scope: { account_id: "work", limit: 100, generate_topics: false },
        report: {
          attempted: 10,
          processed: 10,
          remaining: 10,
          skipped_budget: 0,
          next_cursor: { after_id: 10, model_id: "test-model-v2", account_id: "work", generate_topics: false },
        },
      },
    }));
    await vmContext.refreshStatus();
    assert.equal(pauseBtn.hidden, true, "Pause button hidden when paused");
    assert.equal(indexBtn.disabled, false, "Index button enabled when paused");
    assert.equal(indexBtn.textContent, "Resume");
    assert.equal(statusSpan.textContent, "Meaning indexing paused (work)");

    // Resuming on account "work" sends the stored after_cursor
    indexBtn.click();
    await new Promise((r) => setImmediate(r));
    const enrichCalls = fetchCalls.filter((c) => c.url === "/api/enrich");
    assert(enrichCalls.length >= 1);
    const lastEnrichBody = JSON.parse(enrichCalls[enrichCalls.length - 1].options.body);
    assert.equal(lastEnrichBody.account_id, "work");
    assert.equal(lastEnrichBody.after_cursor?.after_id, 10);

    // Cross-account Resume POST regression: switching to account "other" must not inherit "work"'s cursor
    elements.get("account").value = "other";
    elements.get("account").dispatchEvent({ type: "change" });
    await new Promise((r) => setImmediate(r));
    assert.equal(vm.runInContext("state.enrichCursor", vmContext), null, "Account change must clear enrichCursor");

    indexBtn.click();
    await new Promise((r) => setImmediate(r));
    const crossAccountEnrichCalls = fetchCalls.filter((c) => c.url === "/api/enrich");
    const lastCrossAccountBody = JSON.parse(crossAccountEnrichCalls[crossAccountEnrichCalls.length - 1].options.body);
    assert.equal(lastCrossAccountBody.account_id, "other");
    assert.equal(lastCrossAccountBody.after_cursor, undefined, "Different account must never inherit another account's cursor");

    // Paused same-account job with incompatible cursor (e.g. old model) must clear cursor and start fresh batch
    elements.get("account").value = "work";
    env.setStatusHandler(() => ({
      conversations: 5,
      messages: 20,
      sources: [{ source_id: "chatgpt", account_id: "work" }],
      semantic: {
        configured: true,
        model_id: "test-model-v2",
        coverage: { account_id: "work", eligible: 20, current: 10, pending: 10, legacy: 0 },
      },
      semantic_job: {
        job_id: "job-1234",
        state: "paused",
        scope: { account_id: "work", limit: 100, generate_topics: false },
        report: {
          attempted: 10,
          processed: 10,
          remaining: 10,
          skipped_budget: 0,
          next_cursor: { after_id: 10, model_id: "old-incompatible-model", account_id: "work", generate_topics: false },
        },
      },
    }));
    await vmContext.refreshStatus();
    assert.equal(vm.runInContext("state.enrichCursor", vmContext), null, "Incompatible cursor must be cleared");
    assert.equal(indexBtn.textContent, "Index next batch", "Paused job without compatible cursor must start fresh batch, not resume");
    indexBtn.click();
    await new Promise((r) => setImmediate(r));
    const freshBatchCalls = fetchCalls.filter((c) => c.url === "/api/enrich");
    const lastFreshBatch = JSON.parse(freshBatchCalls[freshBatchCalls.length - 1].options.body);
    assert.equal(lastFreshBatch.account_id, "work");
    assert.equal(lastFreshBatch.after_cursor, undefined, "Fresh batch must not send incompatible after_cursor");

    // Deferred A/B status race: late A response must not overwrite B coverage/job/cursor
    let resolveDeferredA;
    const deferredAPromise = new Promise((resolve) => { resolveDeferredA = resolve; });
    env.setStatusHandler(async (url) => {
      if (url.includes("account_id=work")) {
        return deferredAPromise;
      }
      return {
        conversations: 3,
        messages: 15,
        sources: [{ source_id: "chatgpt", account_id: "work" }, { source_id: "chatgpt", account_id: "other" }],
        semantic: {
          configured: true,
          model_id: "test-model-v2",
          coverage: { account_id: "other", eligible: 15, current: 15, pending: 0, legacy: 0 },
        },
        semantic_job: { state: "idle" },
      };
    });

    elements.get("account").value = "work";
    const pendingWorkStatus = vmContext.refreshStatus();

    // User switches to account "other" while request for "work" is still pending
    elements.get("account").value = "other";
    await vmContext.refreshStatus();
    assert.equal(statusSpan.textContent, "15 messages ready for meaning search");
    assert.equal(vm.runInContext("state.enrichCursor", vmContext), null);

    // Late "work" status arrives with old paused job and cursor
    resolveDeferredA({
      conversations: 5,
      messages: 20,
      sources: [{ source_id: "chatgpt", account_id: "work" }, { source_id: "chatgpt", account_id: "other" }],
      semantic: {
        configured: true,
        model_id: "test-model-v2",
        coverage: { account_id: "work", eligible: 20, current: 2, pending: 18, legacy: 0 },
      },
      semantic_job: {
        job_id: "stale-job-A",
        state: "paused",
        scope: { account_id: "work", limit: 100, generate_topics: false },
        report: {
          next_cursor: { after_id: 2, model_id: "test-model-v2", account_id: "work", generate_topics: false },
        },
      },
    });
    await pendingWorkStatus;

    // Late A status must be ignored; B's status and null cursor must remain intact
    assert.equal(statusSpan.textContent, "15 messages ready for meaning search");
    assert.equal(vm.runInContext("state.enrichCursor", vmContext), null);

    // Pass exhausted with remaining > 0 shows "Some passages need attention" and button "Retry remaining"
    elements.get("account").value = "work";
    env.setStatusHandler(() => ({
      conversations: 5,
      messages: 20,
      sources: [{ source_id: "chatgpt", account_id: "work" }],
      semantic: {
        configured: true,
        model_id: "test-model-v2",
        coverage: { account_id: "work", eligible: 20, current: 15, pending: 5, legacy: 0 },
      },
      semantic_job: {
        job_id: "job-5678",
        state: "complete",
        scope: { account_id: "work", limit: 100, generate_topics: false },
        report: {
          attempted: 5,
          processed: 0,
          remaining: 5,
          skipped_budget: 5,
          pass_exhausted: true,
          next_cursor: null,
        },
      },
    }));
    await vmContext.refreshStatus();
    assert.equal(statusSpan.textContent, "Some passages need attention");
    assert.equal(indexBtn.textContent, "Retry remaining");

    // Clicking "Retry remaining" resets cursor and sends explicit request
    indexBtn.click();
    await new Promise((r) => setImmediate(r));
    const retryCalls = fetchCalls.filter((c) => c.url === "/api/enrich");
    const lastRetryBody = JSON.parse(retryCalls[retryCalls.length - 1].options.body);
    assert.equal(lastRetryBody.after_cursor, undefined, "Retry remaining must clear cursor");

    // Old backend without enrich_controls_version >= 1 disables scoped indexing with explanation
    const oldEnv = await createTestEnv({
      config: { token: "token", semantic_available: true, enrich_controls_version: 0 },
      status: {
        conversations: 2,
        messages: 10,
        sources: [{ source_id: "chatgpt", account_id: "legacy-acc" }],
      },
    });
    const oldIndexBtn = oldEnv.elements.get("index-semantics");
    const oldAccount = oldEnv.elements.get("account");
    oldAccount.value = "legacy-acc";
    await oldEnv.vmContext.refreshStatus();
    assert.equal(oldIndexBtn.disabled, true, "Account-scoped indexing must be disabled on old backend");
    assert(oldIndexBtn.title.includes("Account-scoped indexing is not supported"));
  }
}

console.log("Smart dialogue offline behavior suite passed (activation, visibility, focus, Unicode anchor, partial coverage, keyword fallback, smart pause, exact match focus, explicit submit, dialogue preview, coverage controls).");
