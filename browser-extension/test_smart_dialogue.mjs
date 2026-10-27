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
    "import-account", "import-report", "confirm-import", "mode-note",
  ];
  const hiddenInitialIds = new Set([
    "pair-panel", "index-semantics", "cloud-option", "notice", "pagination", "context-content",
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
    if (urlStr === "/api/status") {
      const statusData = {
        conversations: 1,
        messages: 10,
        sources: [],
        ...(options.status || {}),
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
  // Wait for start() microtasks and initial searches to complete
  for (let i = 0; i < 10; i++) {
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

  // Per-query partial coverage and purpose/example retention in smart mode:
  vm.runInContext("state.mode = 'smart';", vmContext);
  vmContext.renderResults({
    results: [{ title: "R1", text: "text 1", match_type: "meaning" }],
    total: 1,
    semantic: {
      used: true,
      indexed_messages: 120,
      pending_messages: 30,
    },
  });
  const modeNote = elements.get("mode-note").textContent;
  assert(modeNote.includes("related ideas when wording differs"), "Smart mode note must retain purpose description");
  assert(modeNote.includes("“fix wifi” finds “network troubleshooting”"), "Smart mode note must retain generic example");
  assert(modeNote.includes("Meaning coverage is partial: 120 indexed, 30 pending"), "Smart mode note must show per-query partial coverage");

  // Provider-error fallback: semantic.used is explicitly false while retaining purpose/example
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
  assert(fallbackNote.includes("related ideas when wording differs"), "Keyword fallback note must retain purpose description");
  assert(fallbackNote.includes("“fix wifi” finds “network troubleshooting”"), "Keyword fallback note must retain generic example");

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
  assert(notConfiguredNote.includes("related ideas when wording differs"), "Must retain purpose description");
  assert(notConfiguredNote.includes("“fix wifi” finds “network troubleshooting”"), "Must retain generic example");

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
  assert(pendingNote.includes("related ideas when wording differs"), "Must retain purpose description");
  assert(pendingNote.includes("“fix wifi” finds “network troubleshooting”"), "Must retain generic example");

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
  assert(providerErrorNote.includes("related ideas when wording differs"), "Must retain purpose description");
  assert(providerErrorNote.includes("“fix wifi” finds “network troubleshooting”"), "Must retain generic example");

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

  // Assert actual mode-note after real renderResults success and fallback includes idea/wording example:
  assert(modeNote.includes("related ideas when wording differs"), "Mode note on success must include related ideas explanation");
  assert(modeNote.includes("“fix wifi” finds “network troubleshooting”"), "Mode note on success must include generic example");
  assert(fallbackNote.includes("related ideas when wording differs"), "Mode note on fallback must include related ideas explanation");
  assert(fallbackNote.includes("“fix wifi” finds “network troubleshooting”"), "Mode note on fallback must include generic example");
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

console.log("Smart dialogue offline behavior suite passed (activation, visibility, focus, Unicode anchor, partial coverage, keyword fallback, smart pause).");
