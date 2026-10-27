"use strict";

const $ = (id) => document.getElementById(id);
const state = { query: "", mode: "exact", offset: 0, limit: 12, total: 0, results: [], selected: null, token: "", request: 0, contextRequest: 0, semanticAvailable: false, cloudAvailable: false };
let searchTimer;
let semanticTimer;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function notice(text, error = false) {
  $("notice").textContent = text;
  $("notice").classList.toggle("error", error);
  $("notice").hidden = !text;
}

async function request(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", cache: "no-store", ...options });
  let data;
  try { data = await response.json(); } catch { throw new Error(`The local service returned an unreadable response (${response.status}).`); }
  if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : data.error?.message || `Request failed (${response.status}).`);
  return data;
}

function safeUrl(value) {
  if (!value) return null;
  try { const url = new URL(value); return url.protocol === "https:" ? url.href : null; } catch { return null; }
}

function readableDate(value) {
  if (value === null || value === undefined || value === "") return "Date unavailable";
  const date = new Date(typeof value === "number" ? value * (value < 100000000000 ? 1000 : 1) : value);
  return Number.isNaN(date.getTime()) ? "Date unavailable" : date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

// Offsets from Python are Unicode code points, not JavaScript UTF-16 indices.
// Slicing Array.from preserves astral characters. Overlaps merge visually only;
// the source match_count remains the exhaustive count returned by the engine.
function appendHighlighted(target, original, ranges = [], excerpt = false, focusOffset = 0, addAnchor = false) {
  const chars = Array.from(original || "");
  const valid = ranges.filter((range) => Array.isArray(range) && Number.isInteger(range[0]) && Number.isInteger(range[1]) && range[0] >= 0 && range[1] > range[0] && range[1] <= chars.length).sort((a, b) => a[0] - b[0]);
  const semanticFocus = Number.isSafeInteger(focusOffset) ? Math.max(0, Math.min(Math.max(0, chars.length - 1), focusOffset)) : 0;
  const focus = valid[0]?.[0] ?? semanticFocus;
  const from = excerpt ? Math.max(0, focus - 65) : 0;
  const to = excerpt ? Math.min(chars.length, from + 280) : chars.length;
  const merged = [];
  for (const range of valid) {
    const start = Math.max(from, range[0]);
    const end = Math.min(to, range[1]);
    if (start >= end) continue;
    const last = merged[merged.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(last[1], end); else merged.push([start, end]);
  }
  if (from) target.append(document.createTextNode("…"));
  let cursor = from;
  if (!excerpt && addAnchor && Number.isSafeInteger(focusOffset) && focusOffset >= 0 && focusOffset <= chars.length && merged.length === 0) {
    const anchorPoint = focusOffset;
    target.append(document.createTextNode(chars.slice(cursor, anchorPoint).join("")));
    const anchor = element("span", "semantic-anchor");
    anchor.setAttribute("aria-hidden", "true");
    target.append(anchor);
    cursor = anchorPoint;
  } else {
    for (const [start, end] of merged) {
      target.append(document.createTextNode(chars.slice(cursor, start).join("")));
      target.append(element("mark", "", chars.slice(start, end).join("")));
      cursor = end;
    }
  }
  target.append(document.createTextNode(chars.slice(cursor, to).join("")));
  if (to < chars.length) target.append(document.createTextNode("…"));
}

function emptyState(title, description, examples = false) {
  const box = element("div", "empty-state");
  box.append(element("div", "empty-symbol", "“"), element("h3", "", title), element("p", "", description));
  if (examples) {
    const suggestions = element("div", "suggestions");
    for (const query of ["recovery key", "ERR_CFG_204", "中文"]) {
      const button = element("button", "", query);
      button.type = "button";
      button.addEventListener("click", () => { $("query").value = query; runSearch(true); });
      suggestions.append(button);
    }
    box.append(suggestions);
  }
  $("results").replaceChildren(box);
}

function semanticText(semantic, totalMessages) {
  if (!semantic || typeof semantic !== "object") return "Semantic indexing not configured";
  if (Array.isArray(semantic.models)) {
    if (!semantic.configured || !semantic.model_id) return "Meaning search not configured · keyword search available";
    const selected = semantic.models.find((model) => model.model_id === semantic.model_id);
    const indexed = Number(selected?.indexed_messages || 0);
    const vectors = Number(selected?.vectors || 0);
    if (indexed > 0 && vectors > 0) {
      const total = Number(totalMessages || 0);
      if (total > 0 && indexed < total) {
        return `${indexed.toLocaleString()} of ${total.toLocaleString()} messages ready for meaning search (partial)`;
      }
      return `${indexed.toLocaleString()} messages ready for meaning search`;
    }
    return "Meaning search not ready · index the configured model";
  }
  if (semantic.available === true || semantic.ready === true || semantic.enabled === true && semantic.indexed > 0) {
    const count = semantic.indexed ?? semantic.embeddings;
    if (count !== undefined) {
      const total = Number(totalMessages || 0);
      if (total > 0 && Number(count) < total) {
        return `${Number(count).toLocaleString()} of ${total.toLocaleString()} semantic passages indexed (partial)`;
      }
      return `${Number(count).toLocaleString()} semantic passages indexed`;
    }
    return "Meaning search ready";
  }
  return semantic.reason || semantic.message || "Meaning search not ready · keyword fallback available";
}

function semanticJobText(job, semantic, totalMessages) {
  if (job?.state === "running") {
    return "Background meaning indexing in progress";
  }
  if (job?.state === "partial" || job?.state === "failed") {
    const first = job?.report?.errors?.[0];
    let raw = "";
    if (first !== undefined && first !== null) {
      if (typeof first === "object" && typeof first.error === "string") {
        raw = first.error;
      } else if (typeof first === "string") {
        raw = first;
      } else if (typeof first === "object" && first.error !== undefined && first.error !== null) {
        raw = String(first.error);
      } else {
        raw = String(first);
      }
    } else if (typeof job?.error === "string") {
      raw = job.error;
    } else if (job?.error !== undefined && job?.error !== null) {
      raw = String(job.error);
    } else {
      raw = "unknown error";
    }
    const diagnostic = Array.from(String(raw).trim() || "unknown error").slice(0, 240).join("");
    if (job.state === "partial") {
      return `Partial meaning indexing: ${diagnostic}`;
    }
    return `Meaning indexing needs attention: ${diagnostic}`;
  }
  return semanticText(semantic, totalMessages);
}

async function refreshStatus() {
  const status = await request("/api/status");
  const conversations = Number(status.conversations || 0), messages = Number(status.messages || 0);
  $("archive-count").textContent = `${conversations.toLocaleString()} conversations · ${messages.toLocaleString()} original messages`;
  const job = status.semantic_job || {};
  const semantic = status.semantic || status.semantic_index;
  $("semantic-status").textContent = semanticJobText(job, semantic, messages);
  const running = job.state === "running";
  $("index-semantics").hidden = !state.semanticAvailable;
  $("index-semantics").disabled = !state.semanticAvailable || running;
  $("index-semantics").textContent = running
    ? "Indexing meaning…"
    : (job.state === "partial" || job.state === "failed")
    ? "Retry indexing"
    : job.report?.remaining > 0
    ? "Index next batch"
    : "Index meaning";
  clearTimeout(semanticTimer);
  if (running) semanticTimer = setTimeout(() => refreshStatus().catch((error) => notice(error.message, true)), 1800);
  const previous = $("account").value;
  const sources = Array.isArray(status.sources) ? status.sources : [];
  const accounts = [...new Set(sources.map((source) => typeof source === "object" ? source.account_id : null).filter(Boolean))];
  $("account").replaceChildren(new Option("All accounts", ""), ...accounts.map((account) => new Option(account, account)));
  if (accounts.includes(previous)) $("account").value = previous;
  if (!messages && !$("query").value.trim()) emptyState("Your archive starts here.", "Import an extracted ChatGPT JSON export. Your original messages stay on this computer; repeated imports update the same archive.");
  return status;
}

function resetContextViewer() {
  ++state.contextRequest;
  state.selected = null;
  $("context-placeholder").hidden = false;
  $("context-content").hidden = true;
  $("context-title").textContent = "";
  $("context-meta").textContent = "";
  $("original-link").hidden = true;
  $("original-link").removeAttribute("href");
  $("messages").replaceChildren();
  $("context-pane").classList.remove("mobile-open");
  document.querySelectorAll(".result-card").forEach((card) => {
    card.classList.remove("selected", "loading", "error");
    card.removeAttribute("aria-current");
  });
}

function renderResults(data) {
  resetContextViewer();
  state.results = Array.isArray(data.results) ? data.results : [];
  state.total = Number(data.total || 0);
  $("results-title").textContent = state.total ? (state.mode === "exact" ? "Matching messages" : "Relevant passages") : "No matching messages";
  const occurrences = state.results.reduce((count, result) => count + Number(result.match_count || 0), 0);
  $("result-count").textContent = `${state.total.toLocaleString()} ${state.total === 1 ? "message" : "messages"}${state.mode === "exact" && state.results.length ? ` · ${occurrences} occurrences on this page` : ""}`;
  if (state.mode === "smart") {
    const semantic = data.semantic || {};
    const meaningUsed = semantic.used === true;
    if (meaningUsed) {
      let note = "Smart search finds related ideas when wording differs (e.g., “fix wifi” finds “network troubleshooting”). Combines original wording with meaning candidates.";
      if (semantic.pending_messages !== undefined || semantic.indexed_messages !== undefined) {
        const pending = Number(semantic.pending_messages || 0);
        const indexed = Number(semantic.indexed_messages || 0);
        if (pending > 0) {
          note += ` Meaning coverage is partial: ${indexed.toLocaleString()} indexed, ${pending.toLocaleString()} pending.`;
        } else if (indexed > 0) {
          note += ` (${indexed.toLocaleString()} indexed messages).`;
        }
      }
      note += " Every result opens its source dialogue.";
      if (semantic.reranking?.used) note += ` Cloud reranking used ${semantic.reranking.model || "the configured model"}.`;
      else if (semantic.reranking?.error) note += ` Cloud reranking unavailable; local ranking retained. ${semantic.reranking.error}`;
      $("mode-note").textContent = note;
    } else {
      const rawError = (Array.isArray(semantic.errors) && semantic.errors.length ? semantic.errors[0] : null) ||
        semantic.error || semantic.reason || semantic.message;
      let defaultReason = "embedding provider unavailable";
      if (semantic.status === "not_configured") {
        defaultReason = "meaning search not configured";
      } else if (semantic.status === "index_pending") {
        defaultReason = "meaning index pending";
      } else if (semantic.status === "provider_error") {
        defaultReason = "embedding provider unavailable";
      }
      const fallbackReason = rawError ? String(rawError).slice(0, 300) : defaultReason;
      let note = `Keyword fallback is active (${fallbackReason}); meaning search was not used for these results. Smart search finds related ideas when wording differs (e.g., “fix wifi” finds “network troubleshooting”).`;
      if (semantic.pending_messages !== undefined || semantic.indexed_messages !== undefined) {
        const pending = Number(semantic.pending_messages || 0);
        const indexed = Number(semantic.indexed_messages || 0);
        if (pending > 0) {
          note += ` Meaning coverage: ${indexed.toLocaleString()} indexed, ${pending.toLocaleString()} pending (partial).`;
        }
      }
      if (semantic.reranking?.used) note += ` Cloud reranking used ${semantic.reranking.model || "the configured model"}.`;
      else if (semantic.reranking?.error) note += ` Cloud reranking unavailable; local ranking retained. ${semantic.reranking.error}`;
      $("mode-note").textContent = note;
    }
  }
  if (!state.results.length) {
    emptyState("No match in this archive.", "Try fewer words, switch to Smart search, or check the account and speaker filters. New or missing history may still need to be imported.");
  } else {
    const fragment = document.createDocumentFragment();
    state.results.forEach((result, index) => {
      const card = element("article", "result-card");
      card.dataset.index = String(index);
      const metadata = element("div", "card-meta");
      metadata.append(element("span", "source-pill", result.source_id || "ChatGPT"), element("span", "", result.role === "user" ? "You" : "AI reply"));
      const date = element("time", "", readableDate(result.created_at));
      metadata.append(date);
      const title = element("button", "result-title", result.title || "Untitled conversation");
      title.type = "button";
      title.addEventListener("click", () => openContext(result, index));
      const excerpt = element("p", "result-excerpt");
      appendHighlighted(excerpt, result.text || result.snippet || "", result.matches || [], true, result.focus_offset);
      const footer = element("div", "result-footer");
      const accountSpan = element("span", "account-label", result.account_id || "default");
      const actions = element("div", "card-actions");
      const matchLabel = result.match_type === "exact" ? `${result.match_count || 0} ${result.match_count === 1 ? "match" : "matches"}` : `${result.match_type || "ranked"}`;
      actions.append(element("span", "match-info", matchLabel));
      const openBtn = element("button", "open-conversation-btn quiet-button", "Open conversation");
      openBtn.type = "button";
      openBtn.addEventListener("click", () => openContext(result, index));
      actions.append(openBtn);
      footer.append(accountSpan, actions);

      card.addEventListener("click", (event) => {
        if (event.target.closest("button, a, input, select, textarea")) return;
        const selection = window.getSelection ? window.getSelection().toString() : "";
        if (selection) return;
        openContext(result, index);
      });

      card.append(metadata, title, excerpt, footer);
      fragment.append(card);
    });
    $("results").replaceChildren(fragment);
  }
  $("pagination").hidden = state.total <= state.limit;
  $("previous-page").disabled = state.offset === 0;
  $("next-page").disabled = state.offset + state.limit >= state.total;
  $("page-label").textContent = `${Math.floor(state.offset / state.limit) + 1} / ${Math.max(1, Math.ceil(state.total / state.limit))}`;
}

async function runSearch(reset = true) {
  clearTimeout(searchTimer);
  const query = $("query").value.trim();
  state.query = query;
  const checkedMode = document.querySelector('input[name="mode"]:checked')?.value || "exact";
  state.mode = state.semanticAvailable ? checkedMode : "exact";
  if (!state.semanticAvailable) {
    const smartRadio = document.querySelector('input[name="mode"][value="smart"]');
    const exactRadio = document.querySelector('input[name="mode"][value="exact"]');
    if (smartRadio) { smartRadio.checked = false; smartRadio.disabled = true; }
    if (exactRadio) { exactRadio.checked = true; exactRadio.disabled = false; }
  }
  if (reset) state.offset = 0;
  resetContextViewer();
  const generation = ++state.request;
  if (!query) {
    state.total = 0; state.results = [];
    $("results-title").textContent = "Start with something you remember";
    $("result-count").textContent = "";
    $("pagination").hidden = true;
    $("results").setAttribute("aria-busy", "false");
    emptyState("A sentence can bring it all back.", "Try a distinctive word or phrase from a conversation. Search covers only successfully imported messages.", true);
    return;
  }
  const params = new URLSearchParams({ q: query, mode: state.mode, limit: state.limit, offset: state.offset, case_sensitive: $("case-sensitive").checked ? "1" : "0" });
  if ($("role").value) params.set("role", $("role").value);
  if ($("account").value) params.set("account_id", $("account").value);
  if (state.mode === "smart" && state.cloudAvailable && $("cloud-rerank").checked) params.set("cloud", "1");
  $("results").setAttribute("aria-busy", "true");
  $("results-title").textContent = "Searching your archive…";
  try {
    const data = await request(`/api/search?${params}`);
    if (generation !== state.request) return;
    notice("");
    renderResults(data);
    const url = new URL(location.href);
    url.search = new URLSearchParams({ q: query, mode: state.mode }).toString();
    history.replaceState(null, "", url);
  } catch (error) {
    if (generation === state.request) { notice(error.message, true); $("results-title").textContent = "Search unavailable"; }
  } finally { if (generation === state.request) $("results").setAttribute("aria-busy", "false"); }
}

async function openContext(result, index) {
  const generation = ++state.contextRequest;
  state.selected = result;
  document.querySelectorAll(".result-card").forEach((card, i) => {
    const isSelected = i === index;
    card.classList.toggle("selected", isSelected);
    card.classList.toggle("loading", isSelected);
    card.classList.remove("error");
    if (isSelected) card.setAttribute("aria-current", "true");
    else card.removeAttribute("aria-current");
  });
  $("context-placeholder").hidden = true;
  $("context-content").hidden = false;
  $("context-title").textContent = result.title || "Untitled conversation";
  $("context-meta").textContent = `${result.source_id || "chatgpt"} · ${result.account_id || "default"}`;
  $("original-link").hidden = true;
  $("messages").replaceChildren(element("div", "loading", "Loading the original exchange…"));
  $("context-pane").classList.add("mobile-open");
  if (typeof $("context-pane").scrollIntoView === "function") {
    $("context-pane").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  if (typeof $("context-pane").focus === "function") {
    $("context-pane").focus();
  }
  const params = new URLSearchParams({
    conversation_id: result.conversation_id,
    message_id: result.message_id,
    source_id: result.source_id || "chatgpt",
    account_id: result.account_id || "default",
  });
  try {
    const data = await request(`/api/context?${params}`);
    if (generation !== state.contextRequest) return;
    const selectedCard = document.querySelector(`.result-card[data-index="${index}"]`);
    if (selectedCard) selectedCard.classList.remove("loading");
    $("context-title").textContent = data.title || result.title || "Untitled conversation";
    const original = safeUrl(data.url || result.url);
    $("original-link").hidden = !original;
    if (original) $("original-link").href = original;
    const fragment = document.createDocumentFragment();
    const isSemanticOnly = result.match_type === "semantic" || (!result.matches || result.matches.length === 0);
    for (const message of data.messages || []) {
      const matched = String(message.id) === String(result.message_id);
      const section = element("section", `message ${message.role === "user" ? "user" : "assistant"}${matched ? " matched" : ""}`);
      const label = element("div", "message-label");
      label.append(element("span", "avatar", message.role === "user" ? "Y" : "✳"), element("span", "", message.role === "user" ? "You" : "Assistant"));
      if (matched) label.append(element("span", "match-label", "Matched message"));
      const text = element("div", "message-text");
      const messageCharsLength = Array.from(message.text || "").length;
      const validOffset = Number.isSafeInteger(result.focus_offset) && result.focus_offset >= 0 && result.focus_offset <= messageCharsLength;
      const shouldAnchor = matched && isSemanticOnly && validOffset;
      appendHighlighted(
        text,
        message.text || "",
        matched ? result.matches || [] : [],
        false,
        matched ? result.focus_offset : 0,
        shouldAnchor
      );
      section.append(label, text);
      fragment.append(section);
    }
    $("messages").replaceChildren(fragment);
    const matched = $("messages").querySelector(".matched");
    const anchor = matched?.querySelector(".semantic-anchor");
    if (anchor) {
      if (typeof anchor.scrollIntoView === "function") {
        anchor.scrollIntoView({ block: "center" });
      } else if (typeof anchor.offsetTop === "number" && anchor.offsetTop > 0) {
        $("messages").scrollTop = Math.max(0, anchor.offsetTop - $("messages").offsetTop - 25);
      } else if (matched) {
        $("messages").scrollTop = Math.max(0, matched.offsetTop - $("messages").offsetTop - 25);
      }
    } else if (matched) {
      if (typeof matched.scrollIntoView === "function") {
        matched.scrollIntoView({ block: "start" });
      }
      $("messages").scrollTop = Math.max(0, matched.offsetTop - $("messages").offsetTop - 25);
    }
  } catch (error) {
    if (generation === state.contextRequest) {
      const selectedCard = document.querySelector(`.result-card[data-index="${index}"]`);
      if (selectedCard) {
        selectedCard.classList.remove("loading");
        selectedCard.classList.add("error");
      }
      $("messages").replaceChildren(element("p", "loading error-text", error.message));
    }
  }
}

function closeContext() {
  resetContextViewer();
}

async function copyText(text, button, confirmation) {
  try { await navigator.clipboard.writeText(text); const old = button.textContent; button.textContent = confirmation; setTimeout(() => { button.textContent = old; }, 1800); }
  catch { notice("Clipboard access is unavailable. Select and copy the text directly.", true); }
}

$("search-form").addEventListener("submit", (event) => { event.preventDefault(); runSearch(true); });
$("query").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => runSearch(true), 250); });
document.querySelectorAll('input[name="mode"]').forEach((input) => input.addEventListener("change", () => {
  if (!state.semanticAvailable) {
    const smartRadio = document.querySelector('input[name="mode"][value="smart"]');
    const exactRadio = document.querySelector('input[name="mode"][value="exact"]');
    if (smartRadio) { smartRadio.checked = false; smartRadio.disabled = true; }
    if (exactRadio) { exactRadio.checked = true; exactRadio.disabled = false; }
    $("case-sensitive").disabled = false;
    $("cloud-rerank").disabled = true;
    $("mode-note").textContent = "Smart search is off · Exact search is available.";
    runSearch(true);
    return;
  }
  const smart = input.value === "smart";
  $("case-sensitive").disabled = smart;
  $("cloud-rerank").disabled = !smart;
  $("mode-note").textContent = smart
    ? "Smart search finds related ideas when wording differs (e.g., “fix wifi” finds “network troubleshooting”). Ranks keyword and available meaning-based candidates."
    : "Exact for remembered words; Smart for related ideas when wording differs (e.g., “fix wifi” finds “network troubleshooting”). Capitalization is ignored unless “Match case” is enabled.";
  runSearch(true);
}));
for (const id of ["role", "account", "case-sensitive", "cloud-rerank"]) $(id).addEventListener("change", () => runSearch(true));
document.querySelectorAll("[data-query]").forEach((button) => button.addEventListener("click", () => { $("query").value = button.dataset.query; runSearch(true); }));
$("previous-page").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); runSearch(false); });
$("next-page").addEventListener("click", () => { state.offset += state.limit; runSearch(false); });
$("close-context").addEventListener("click", closeContext);
$("copy-message").addEventListener("click", () => { if (state.selected) copyText(state.selected.text, $("copy-message"), "Copied"); });
$("pair-button").addEventListener("click", () => { $("pair-panel").hidden = !$("pair-panel").hidden; $("pair-button").setAttribute("aria-expanded", String(!$("pair-panel").hidden)); });
$("copy-token").addEventListener("click", () => copyText(state.token, $("copy-token"), "Copied"));
$("index-semantics").addEventListener("click", async () => {
  if (!state.semanticAvailable) return;
  $("index-semantics").disabled = true;
  try {
    await request("/api/enrich", { method: "POST", headers: { "Content-Type": "application/json", "Authorization": `Bearer ${state.token}` }, body: JSON.stringify({ limit: 100, generate_topics: false }) });
    notice("Meaning indexing started for up to 100 changed passages. No models are downloaded by this action.");
    await refreshStatus();
  } catch (error) { notice(error.message, true); $("index-semantics").disabled = false; }
});
$("import-button").addEventListener("click", () => { $("import-report").textContent = ""; $("import-dialog").showModal(); });
$("close-import").addEventListener("click", () => $("import-dialog").close());
$("import-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("import-file").files[0];
  if (!file) return;
  if (file.size > 64 * 1024 * 1024) { $("import-report").textContent = "This file exceeds 64 MB. Split the export at conversation boundaries before importing."; return; }
  $("confirm-import").disabled = true;
  $("import-report").textContent = "Reading and importing original messages…";
  try {
    const payload = JSON.parse(await file.text());
    const report = await request("/api/import", { method: "POST", headers: { "Content-Type": "application/json", "Authorization": `Bearer ${state.token}` }, body: JSON.stringify({ payload, source_id: "chatgpt", account_id: $("import-account").value.trim() || "default" }) });
    const errors = Array.isArray(report.errors) ? report.errors : [];
    $("import-report").textContent = `${report.conversations || 0} conversations · ${report.messages || 0} messages imported · ${report.skipped || 0} skipped.${errors.length ? `\nPartial import: ${errors.map((error) => typeof error === "string" ? error : JSON.stringify(error)).slice(0, 5).join("; ")}` : "\nReady to search. Import counts describe this file, not complete account coverage."}`;
    await refreshStatus();
    if (state.query) await runSearch(true);
  } catch (error) { $("import-report").textContent = error instanceof SyntaxError ? "This is not valid JSON. Extract the history JSON from your export ZIP first." : error.message; }
  finally { $("confirm-import").disabled = false; }
});

document.addEventListener("keydown", (event) => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "") || document.activeElement?.isContentEditable;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k" || event.key === "/" && !typing) { event.preventDefault(); $("query").focus(); $("query").select(); }
  if (event.key === "Escape" && !$("import-dialog").open) closeContext();
});

(async function start() {
  try {
    const [config, status] = await Promise.all([request("/api/config"), refreshStatus()]);
    state.token = config.token || config.import_token || "";
    state.semanticAvailable = config.semantic_available === true;
    state.cloudAvailable = config.cloud_rerank_available === true;
    const smartRadio = document.querySelector('input[name="mode"][value="smart"]');
    const exactRadio = document.querySelector('input[name="mode"][value="exact"]');
    if (!state.semanticAvailable) {
      if (smartRadio) {
        smartRadio.disabled = true;
        smartRadio.checked = false;
      }
      if (exactRadio) {
        exactRadio.checked = true;
        exactRadio.disabled = false;
      }
      $("case-sensitive").disabled = false;
      $("mode-note").textContent = "Smart search is off · Exact search is available.";
    } else {
      if (smartRadio) smartRadio.disabled = false;
    }
    $("index-semantics").hidden = !state.semanticAvailable;
    $("index-semantics").disabled = !state.semanticAvailable || status?.semantic_job?.state === "running";
    $("cloud-option").hidden = !state.cloudAvailable;
    $("cloud-rerank").disabled = true;
    $("pair-endpoint").value = location.origin;
    $("pair-token").value = state.token;
    const params = new URLSearchParams(location.search);
    if (!state.semanticAvailable && params.get("mode") === "smart") {
      const url = new URL(location.href);
      const urlParams = new URLSearchParams(location.search);
      urlParams.set("mode", "exact");
      url.search = urlParams.toString();
      history.replaceState(null, "", url);
    } else if (state.semanticAvailable && params.get("mode") === "smart") {
      if (smartRadio) smartRadio.checked = true;
      if (exactRadio) exactRadio.checked = false;
      $("case-sensitive").disabled = true;
      $("cloud-rerank").disabled = false;
      $("mode-note").textContent = "Smart search finds related ideas when wording differs (e.g., “fix wifi” finds “network troubleshooting”). Ranks keyword and available meaning-based candidates.";
    }
    if (params.get("q")) { $("query").value = params.get("q"); await runSearch(true); }
  } catch (error) { notice(`Cannot reach the local archive. ${error.message}`, true); $("archive-count").textContent = "Archive unavailable"; $("semantic-status").textContent = ""; }
})();
