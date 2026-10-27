"use strict";
importScripts("adapter.js");

const adapter = globalThis.ChatRecallAdapter;
const DEFAULTS = { endpoint: "http://127.0.0.1:8765", token: "", workspaceId: "", automatic: false };
const LIMIT = 20;
const MAX_PAGES = 3;
let running = false;
let initPromise = null;

async function settings() { const saved = await chrome.storage.local.get("settings"); return { ...DEFAULTS, ...(saved.settings || {}) }; }
async function setStatus(title, detail, extra = {}) { await chrome.storage.local.set({ status: { title, detail, updated: new Date().toISOString(), ...extra } }); }

async function getJob() {
  const saved = await chrome.storage.local.get("syncJob");
  const job = saved.syncJob;
  if (job && typeof job === "object" && job.version === 1) {
    return {
      version: 1,
      enabled: job.enabled === true,
      state: typeof job.state === "string" ? job.state : "paused",
      nextAttemptAt: (typeof job.nextAttemptAt === "number" && Number.isFinite(job.nextAttemptAt)) ? job.nextAttemptAt : null,
      attempt: (typeof job.attempt === "number" && Number.isInteger(job.attempt) && job.attempt >= 0) ? job.attempt : 0,
      revision: (typeof job.revision === "number" && Number.isInteger(job.revision) && job.revision >= 0) ? job.revision : 1,
      accountId: typeof job.accountId === "string" && job.accountId ? job.accountId : null,
      ...(job.blocked ? { blocked: true } : {})
    };
  }
  const cfg = await settings();
  return {
    version: 1,
    enabled: cfg.automatic === true,
    state: cfg.automatic ? "waiting" : "paused",
    nextAttemptAt: null,
    attempt: 0,
    revision: 1,
    accountId: null
  };
}

async function saveJob(job) {
  const safeJob = {
    version: 1,
    enabled: job.enabled === true,
    state: typeof job.state === "string" ? job.state : "paused",
    nextAttemptAt: (typeof job.nextAttemptAt === "number" && Number.isFinite(job.nextAttemptAt)) ? job.nextAttemptAt : null,
    attempt: (typeof job.attempt === "number" && Number.isInteger(job.attempt) && job.attempt >= 0) ? job.attempt : 0,
    revision: (typeof job.revision === "number" && Number.isInteger(job.revision) && job.revision >= 0) ? job.revision : 1,
    accountId: typeof job.accountId === "string" && job.accountId ? job.accountId : null,
    ...(job.blocked ? { blocked: true } : {})
  };
  await chrome.storage.local.set({ syncJob: safeJob });
  return safeJob;
}

function ensureInitialized() {
  if (!initPromise) {
    initPromise = reconcileJobAndAlarms();
  }
  return initPromise;
}

async function reconcileJobAndAlarms() {
  if (running) return;
  const cfg = await settings();
  if (running) return;
  let job = await getJob();
  if (running) return;
  const now = Date.now();

  if (job.state === "running") {
    if (cfg.automatic) {
      job.state = "waiting";
      job.nextAttemptAt = now;
      await saveJob(job);
      await chrome.alarms.create("chat-recall-sync", { when: now });
    } else {
      job.state = "paused";
      await saveJob(job);
    }
    return;
  }

  if (cfg.automatic) {
    job.enabled = true;
    if (job.state === "complete") {
      await chrome.alarms.clear("chat-recall-sync");
      return;
    }
    if (job.state === "attention" && job.blocked) {
      await chrome.alarms.clear("chat-recall-sync");
      return;
    }
    if (job.nextAttemptAt && now < job.nextAttemptAt) {
      job.state = "cooldown";
      await saveJob(job);
      let existing = null;
      try {
        if (typeof chrome.alarms.get === "function") existing = await chrome.alarms.get("chat-recall-sync");
      } catch {}
      if (!existing || Math.abs((existing.scheduledTime || 0) - job.nextAttemptAt) > 1000) {
        await chrome.alarms.create("chat-recall-sync", { when: job.nextAttemptAt });
      }
    } else {
      job.state = "waiting";
      job.nextAttemptAt = null;
      await saveJob(job);
      let existing = null;
      try {
        if (typeof chrome.alarms.get === "function") existing = await chrome.alarms.get("chat-recall-sync");
      } catch {}
      if (!existing || (existing.scheduledTime || 0) > now + 1000) {
        await chrome.alarms.create("chat-recall-sync", { when: now });
      }
    }
  } else {
    job.enabled = false;
    if (job.state !== "attention" && job.state !== "complete") {
      job.state = "paused";
    }
    await saveJob(job);
    await chrome.alarms.clear("chat-recall-sync");
  }
}

function computeBackoff(attempt, retryAfterMs, now = Date.now()) {
  const exponent = Math.max(0, attempt - 1);
  const rawDelay = Math.min(30 * 60 * 1000, 60000 * Math.pow(2, exponent));
  const jitter = rawDelay * (Math.random() * 0.20);
  const fallbackDelay = Math.min(30 * 60 * 1000, Math.round(rawDelay + jitter));

  let delay = fallbackDelay;
  if (typeof retryAfterMs === "number" && Number.isFinite(retryAfterMs) && retryAfterMs > 0) {
    delay = Math.max(retryAfterMs, fallbackDelay);
  }
  const nextAttemptAt = now + delay;
  return { delay, nextAttemptAt };
}

async function importLocal(config, payload, accountId, coverage) {
  const response = await fetch(`${adapter.endpoint(config.endpoint)}/api/import`, { method: "POST", headers: { "Content-Type": "application/json", "Authorization": `Bearer ${config.token}` }, body: JSON.stringify({ payload, source_id: "chatgpt", account_id: accountId, coverage }) });
  let report;
  try { report = await response.json(); } catch { throw new Error(`Local service returned an unreadable response (${response.status}). Check that Chat Recall is running.`); }
  if (!response.ok) throw new Error(report.error?.message || report.error || `Local import failed (${response.status}). Check the pairing token and extension-origin allowlist.`);
  return report;
}

// This function executes inside an existing signed-in ChatGPT tab. Access tokens
// exist only in that page execution and are never returned or sent to Chat Recall.
async function readPage(arguments_) {
  let sessionSource = "session-endpoint";

  function parseRetryAfter(response) {
    let headerVal = null;
    try {
      if (response && response.headers) {
        if (typeof response.headers.get === "function") {
          headerVal = response.headers.get("retry-after") || response.headers.get("Retry-After");
        } else if (response.headers["retry-after"]) {
          headerVal = response.headers["retry-after"];
        } else if (response.headers["Retry-After"]) {
          headerVal = response.headers["Retry-After"];
        }
      }
    } catch {}

    if (typeof headerVal !== "string") return { retry_after_ms: null };
    const trimmed = headerVal.trim();
    if (!trimmed) return { retry_after_ms: null };

    // 1. delay-seconds: strictly 1*DIGIT
    if (/^\d+$/.test(trimmed)) {
      if (trimmed.length > 15) {
        return { retry_after_ms: null, retry_after_unrepresentable: true };
      }
      const seconds = Number(trimmed);
      if (!Number.isSafeInteger(seconds) || seconds < 0) {
        return { retry_after_ms: null, retry_after_unrepresentable: true };
      }
      const ms = seconds * 1000;
      if (ms > Number.MAX_SAFE_INTEGER || (Date.now() + ms) > 8640000000000000) {
        return { retry_after_ms: null, retry_after_unrepresentable: true };
      }
      return { retry_after_ms: ms };
    }

    // 2. HTTP-date: IMF-fixdate, RFC 850, or ANSI C asctime
    const MONTHS = { Jan: 0, Feb: 1, Mar: 2, Apr: 3, May: 4, Jun: 5, Jul: 6, Aug: 7, Sep: 8, Oct: 9, Nov: 10, Dec: 11 };

    // IMF-fixdate: Sun, 06 Nov 1994 08:49:37 GMT
    const imfMatch = trimmed.match(/^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(\d{2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\s+GMT$/);
    if (imfMatch) {
      const day = Number(imfMatch[1]);
      const month = MONTHS[imfMatch[2]];
      const year = Number(imfMatch[3]);
      const hour = Number(imfMatch[4]);
      const min = Number(imfMatch[5]);
      const sec = Number(imfMatch[6]);
      if (hour > 23 || min > 59 || sec > 60 || day < 1 || day > 31) {
        return { retry_after_ms: null };
      }
      const targetUtc = Date.UTC(year, month, day, hour, min, sec);
      if (!Number.isFinite(targetUtc) || targetUtc > 8640000000000000) {
        return { retry_after_ms: null, retry_after_unrepresentable: true };
      }
      const diff = targetUtc - Date.now();
      return { retry_after_ms: diff > 0 ? diff : 0 };
    }

    // RFC 850: Sunday, 06-Nov-94 08:49:37 GMT
    const rfc850Match = trimmed.match(/^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(\d{2})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\s+GMT$/);
    if (rfc850Match) {
      const day = Number(rfc850Match[1]);
      const month = MONTHS[rfc850Match[2]];
      const twoDigitYear = Number(rfc850Match[3]);
      const currentYear = new Date().getUTCFullYear();
      let year = Math.floor(currentYear / 100) * 100 + twoDigitYear;
      if (year > currentYear + 50) {
        year -= 100;
      } else if (year <= currentYear - 50) {
        year += 100;
      }
      const hour = Number(rfc850Match[4]);
      const min = Number(rfc850Match[5]);
      const sec = Number(rfc850Match[6]);
      if (hour > 23 || min > 59 || sec > 60 || day < 1 || day > 31) {
        return { retry_after_ms: null };
      }
      const targetUtc = Date.UTC(year, month, day, hour, min, sec);
      if (!Number.isFinite(targetUtc)) return { retry_after_ms: null, retry_after_unrepresentable: true };
      const diff = targetUtc - Date.now();
      return { retry_after_ms: diff > 0 ? diff : 0 };
    }

    // ANSI C asctime(): Sun Nov  6 08:49:37 1994 (RFC 9110 Section 5.6.7: treated as UTC without exception)
    const asctimeMatch = trimmed.match(/^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})\s+(\d{4})$/);
    if (asctimeMatch) {
      const month = MONTHS[asctimeMatch[1]];
      const day = Number(asctimeMatch[2]);
      const hour = Number(asctimeMatch[3]);
      const min = Number(asctimeMatch[4]);
      const sec = Number(asctimeMatch[5]);
      const year = Number(asctimeMatch[6]);
      if (hour > 23 || min > 59 || sec > 60 || day < 1 || day > 31) {
        return { retry_after_ms: null };
      }
      const targetUtc = Date.UTC(year, month, day, hour, min, sec);
      if (!Number.isFinite(targetUtc) || targetUtc > 8640000000000000) {
        return { retry_after_ms: null, retry_after_unrepresentable: true };
      }
      const diff = targetUtc - Date.now();
      return { retry_after_ms: diff > 0 ? diff : 0 };
    }

    return { retry_after_ms: null };
  }

  function classifyHttpError(stage, response, source = sessionSource) {
    const status = response.status;
    let isChallenge = false;
    try {
      const headerVal = response.headers?.get ? response.headers.get("cf-mitigated") : response.headers?.["cf-mitigated"];
      if (typeof headerVal === "string" && headerVal.toLowerCase() === "challenge") {
        isChallenge = true;
      }
    } catch {}

    let retryAfterData = {};
    if (status === 429 && !isChallenge) {
      retryAfterData = parseRetryAfter(response);
    }

    let error;
    if (isChallenge) {
      error = `ChatGPT ${stage} returned ${status} (challenge). Request refused; account may still be signed in. Refresh ChatGPT in your browser to complete any challenge check, then retry manually.`;
    } else if (status === 401) {
      error = `ChatGPT ${stage} returned 401. Authentication required. Sign in again before retrying.`;
    } else if (status === 403) {
      error = `ChatGPT ${stage} returned 403. Request refused; account may still be signed in. Refresh ChatGPT in your browser, then retry manually.`;
    } else if (status === 429) {
      error = `ChatGPT ${stage} returned 429. Rate limited; the checkpoint is preserved. Wait before retrying.`;
    } else {
      error = `ChatGPT ${stage} returned ${status}. The checkpoint is preserved. Retry later.`;
    }
    return {
      error,
      status,
      stage,
      session_source: source,
      ...(isChallenge ? { challenge: true } : {}),
      ...(retryAfterData.retry_after_ms !== undefined && retryAfterData.retry_after_ms !== null ? { retry_after_ms: retryAfterData.retry_after_ms } : {}),
      ...(retryAfterData.retry_after_unrepresentable ? { retry_after_unrepresentable: true } : {})
    };
  }

  function getBootstrapSession() {
    try {
      if (typeof document === "undefined" || !document || typeof document.getElementById !== "function") {
        return null;
      }
      const element = document.getElementById("client-bootstrap");
      if (!element) return null;
      const tagName = element.tagName || element.nodeName;
      if (typeof tagName !== "string" || tagName.toUpperCase() !== "SCRIPT") {
        return null;
      }
      const type = element.type || (typeof element.getAttribute === "function" ? element.getAttribute("type") : null);
      if (typeof type !== "string" || type.toLowerCase() !== "application/json") {
        return null;
      }
      const text = element.textContent;
      if (typeof text !== "string" || text.length > 2 * 1024 * 1024) {
        return null;
      }
      if (new TextEncoder().encode(text).byteLength > 2 * 1024 * 1024) {
        return null;
      }
      let parsed;
      try {
        parsed = JSON.parse(text);
      } catch {
        return null;
      }
      if (!parsed || typeof parsed !== "object") return null;
      const s = parsed.session;
      if (!s || typeof s !== "object") return null;
      if (typeof s.accessToken !== "string" || !s.accessToken.trim()) return null;
      if (typeof s.user?.id !== "string" || !s.user.id.trim()) return null;

      const token = s.accessToken;
      const parts = token.trim().split(".");
      if (parts.length === 3) {
        if (typeof atob !== "function") return null;
        let base64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
        const pad = base64.length % 4;
        if (pad === 2) base64 += "==";
        else if (pad === 3) base64 += "=";
        else if (pad === 1) return null;
        let decoded;
        try {
          decoded = atob(base64);
        } catch {
          return null;
        }
        let payload;
        try {
          payload = JSON.parse(decoded);
        } catch {
          return null;
        }
        if (!payload || typeof payload !== "object") return null;
        if (typeof payload.exp !== "number" || !Number.isFinite(payload.exp)) {
          return null;
        }
        if (payload.exp * 1000 <= Date.now() + 30000) {
          return null;
        }
      }

      return {
        accessToken: token,
        user: { id: s.user.id }
      };
    } catch {
      return null;
    }
  }

  let currentStage = "session";
  try {
    let session = getBootstrapSession();
    if (session) {
      sessionSource = "page-bootstrap";
    } else {
      sessionSource = "session-endpoint";
      const sessionResponse = await fetch("/api/auth/session", { credentials: "include", cache: "no-store" });
      if (!sessionResponse.ok) return classifyHttpError("session", sessionResponse, sessionSource);
      try {
        session = await sessionResponse.json();
      } catch {
        return { error: "ChatGPT session returned unreadable response data. The checkpoint is preserved.", status: sessionResponse.status, stage: "session", session_source: sessionSource };
      }
      if (typeof session?.accessToken !== "string" || !session.accessToken || typeof session?.user?.id !== "string" || !session.user.id) {
        return { error: "No usable signed-in ChatGPT session was found. Open ChatGPT in this browser and sign in.", stage: "session", session_source: sessionSource };
      }
    }
    const identity = `${session.user.id}:${arguments_.workspaceId || "default"}`;
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(identity));
    const accountId = `web-${Array.from(new Uint8Array(digest)).slice(0, 12).map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
    if (arguments_.operation === "identity") return { accountId, session_source: sessionSource };
    const headers = { "Authorization": `Bearer ${session.accessToken}` };
    if (arguments_.workspaceId) headers["ChatGPT-Account-ID"] = arguments_.workspaceId;

    currentStage = "history";
    const listUrl = new URL("/backend-api/conversations", location.origin);
    listUrl.searchParams.set("limit", String(arguments_.limit));
    listUrl.searchParams.set("offset", String(arguments_.offset));
    listUrl.searchParams.set("order", "updated");
    if (arguments_.cursor) listUrl.searchParams.set("cursor", arguments_.cursor);
    const response = await fetch(listUrl, { credentials: "include", headers, cache: "no-store" });
    if (!response.ok) return classifyHttpError("history", response, sessionSource);
    let list;
    try {
      list = await response.json();
    } catch {
      return { error: "ChatGPT history returned unreadable response data. The checkpoint is preserved.", status: response.status, stage: "history", session_source: sessionSource };
    }
    if (!list || !Array.isArray(list.items)) return { error: "The ChatGPT conversation-list format changed. The import stopped without claiming completeness.", stage: "history", session_source: sessionSource };
    const conversations = [];
    const importedVersions = {};
    let unchanged = 0;
    for (const item of list.items) {
      if (typeof item?.id !== "string" || !item.id) return { error: "The ChatGPT list contains a record without an ID. The page checkpoint is preserved.", stage: "history", session_source: sessionSource };
      const time = item.update_time ?? item.create_time;
      const milliseconds = typeof time === "number" ? Math.round(time * (time < 100000000000 ? 1000 : 1)) : Date.parse(time);
      const version = Number.isFinite(milliseconds) ? String(milliseconds) : null;
      if (version && arguments_.versions?.[item.id] === version) { unchanged++; continue; }

      currentStage = "conversation";
      const detail = await fetch(`/backend-api/conversation/${encodeURIComponent(item.id)}`, { credentials: "include", headers, cache: "no-store" });
      if (!detail.ok) {
        const classified = classifyHttpError("conversation", detail, sessionSource);
        if (detail.status === 429 && !classified.challenge) {
          return {
            ...classified,
            partial: true,
            conversations,
            importedVersions,
            accountId
          };
        }
        return classified;
      }
      let conversation;
      try {
        conversation = await detail.json();
      } catch {
        return { error: "ChatGPT conversation returned unreadable response data. The checkpoint is preserved.", status: detail.status, stage: "conversation", session_source: sessionSource };
      }
      if (!conversation || typeof conversation.mapping !== "object" || !conversation.mapping) {
        return { error: "ChatGPT conversation has an unsupported format. No complete-sync claim was recorded.", stage: "conversation", session_source: sessionSource };
      }
      conversations.push({ ...conversation, id: item.id });
      if (version) importedVersions[item.id] = version;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    return { accountId, list, conversations, importedVersions, unchanged, session_source: sessionSource };
  } catch {
    return { error: `ChatGPT ${currentStage} request failed. The checkpoint is preserved.`, stage: currentStage, session_source: sessionSource };
  }
}

async function readChatGPT(config, args) {
  const tabs = await chrome.tabs.query({ url: ["https://chatgpt.com/*", "https://chat.openai.com/*"] });
  const tab = tabs.find((candidate) => candidate.active && !candidate.discarded) || tabs.find((candidate) => !candidate.discarded);
  if (!tab) throw new Error("Open a signed-in ChatGPT tab in this browser. The bridge never opens or signs in to an account for you.");
  const results = await chrome.scripting.executeScript({ target: { tabId: tab.id }, world: "MAIN", func: readPage, args: [{ ...args, workspaceId: config.workspaceId }] });
  const result = results?.[0]?.result;
  if (!result) throw new Error("The signed-in tab did not return a readable result. Refresh that tab and retry.");
  if (result.error) {
    const error = new Error(result.error);
    if (result.status !== undefined) error.status = result.status;
    if (result.stage !== undefined) error.stage = result.stage;
    if (result.challenge !== undefined) error.challenge = result.challenge;
    if (result.session_source !== undefined) error.session_source = result.session_source;
    if (result.retry_after_ms !== undefined) error.retry_after_ms = result.retry_after_ms;
    if (result.retry_after_unrepresentable !== undefined) error.retry_after_unrepresentable = result.retry_after_unrepresentable;
    if (result.partial) {
      error.partial = true;
      error.conversations = result.conversations;
      error.importedVersions = result.importedVersions;
      error.accountId = result.accountId;
    }
    throw error;
  }
  return result;
}

async function syncHistory(options = {}) {
  const isManual = options.manual !== false;
  if (running) return { ok: false, error: "An import is already running. Its checkpoint will be saved before the next run." };
  running = true;
  let imported = 0;
  const observedSources = new Set();
  function formatSourceLabel() {
    if (observedSources.size === 0) return null;
    return Array.from(observedSources).join(", ");
  }

  let config = null;
  let identity = null;
  let key = null;
  let checkpoint = null;
  let versions = null;
  let offset = 0;
  let startRevision = 1;

  try {
    await ensureInitialized();
    config = await settings();
    if (!config.token) throw new Error("Pair your local Chat Recall service first.");
    adapter.endpoint(config.endpoint);

    let job = await getJob();
    startRevision = job.revision;
    const now = Date.now();

    if (job.blocked) {
      return { ok: false, error: "Sync is blocked due to unrepresentable rate limit delay. Manual resolution required.", job };
    }

    if (!isManual) {
      if (!config.automatic || !job.enabled || job.state === "complete" || job.state === "paused" || job.state === "attention") {
        return { ok: false, error: "Automatic sync is not enabled or job is not in a runnable state.", job };
      }
    }

    if (job.nextAttemptAt && now < job.nextAttemptAt) {
      if (job.enabled) {
        await chrome.alarms.create("chat-recall-sync", { when: job.nextAttemptAt });
      }
      return { ok: false, error: "Backoff cooldown active. Remote requests deferred.", nextAttemptAt: job.nextAttemptAt, job };
    }

    job.state = "running";
    await saveJob(job);

    await setStatus("Checking ChatGPT session…", "Only read-only account endpoints are used. Project, archived, and native-app coverage remain unverified.");
    identity = await readChatGPT(config, { operation: "identity" });
    if (identity.session_source) observedSources.add(identity.session_source);

    const postIdentJob = await getJob();
    const postIdentCfg = await settings();
    if (postIdentJob.revision !== startRevision || postIdentCfg.endpoint !== config.endpoint || postIdentCfg.token !== config.token || postIdentCfg.workspaceId !== config.workspaceId) {
      const sourceLabel = formatSourceLabel();
      return { ok: false, error: "Settings or job state changed during identity check.", job: postIdentJob, ...(sourceLabel ? { session_source: sourceLabel } : {}) };
    }

    if (postIdentCfg.automatic) {
      if (!postIdentJob.accountId) {
        postIdentJob.accountId = identity.accountId;
        await saveJob(postIdentJob);
      } else if (postIdentJob.accountId !== identity.accountId) {
        postIdentJob.state = "attention";
        postIdentJob.enabled = false;
        postIdentCfg.automatic = false;
        await chrome.storage.local.set({ settings: postIdentCfg });
        await chrome.alarms.clear("chat-recall-sync");
        await saveJob(postIdentJob);
        throw new Error("Signed-in ChatGPT account changed. Automatic backfill paused for account safety.");
      }
    }

    key = `checkpoint:${identity.accountId}`;
    const saved = await chrome.storage.local.get(key);
    checkpoint = adapter.checkpoint(saved[key]);
    versions = { ...(checkpoint.versions || {}) };
    offset = checkpoint.offset;
    let cursor = checkpoint.cursor;
    let pages = Number(checkpoint.pages || 0);
    let completed = false;

    const maxPagesToRead = config.automatic ? 1 : MAX_PAGES;

    for (let pageNumber = 0; pageNumber < maxPagesToRead; pageNumber++) {
      const curCfg = await settings();
      const curJob = await getJob();
      if (curJob.revision !== startRevision) break;
      if (!curCfg.automatic && !isManual) break;

      const sourceLabel = formatSourceLabel();
      const sourceSuffix = sourceLabel ? ` Session source: ${sourceLabel}.` : "";
      await setStatus("Importing account history…", `Reading ordinary-list page at offset ${offset}. ${imported} changed conversations sent this run.${sourceSuffix}`);

      const data = await readChatGPT(config, { operation: "page", offset, cursor, limit: LIMIT, versions });
      if (data.session_source) observedSources.add(data.session_source);
      if (data.accountId !== identity.accountId) throw new Error("The signed-in account changed during import. The scan stopped without mixing accounts.");

      const postJob = await getJob();
      const postCfg = await settings();
      if (postJob.revision !== startRevision || postCfg.endpoint !== config.endpoint || postCfg.token !== config.token || postCfg.workspaceId !== config.workspaceId) {
        break;
      }

      const page = adapter.page(data.list, offset, LIMIT);
      const coverage = { adapter: "chatgpt-web-experimental", status: "partial", account_id: data.accountId, last_success: new Date().toISOString(), ordinary_list_offset: offset, project_coverage: "unverified", archive_coverage: "unverified", native_app_sync: "unverified", full_account: false, has_missing_conversations: page.missing };

      const report = await importLocal(config, data.conversations, data.accountId, coverage);
      if (Array.isArray(report.errors) && report.errors.length) {
        throw new Error(`Local import was partial (${report.errors.length} errors). The page checkpoint was preserved for retry.`);
      }

      imported += data.conversations.length;
      Object.assign(versions, data.importedVersions);
      pages++;
      const pollCaughtUp = checkpoint.phase === "poll" && page.items.length > 0 && data.unchanged === page.items.length;
      completed = page.exhausted || pollCaughtUp;
      offset += page.items.length;
      if (page.cursor && page.cursor === cursor && !completed) throw new Error("ChatGPT repeated a pagination cursor. The scan stopped rather than loop or claim completeness.");
      cursor = page.cursor;
      await chrome.storage.local.set({ [key]: { offset: completed ? 0 : offset, cursor: completed ? null : cursor, versions, phase: completed ? "poll" : checkpoint.phase, pages, updated: new Date().toISOString() } });
      if (completed) break;

      if (!curCfg.automatic && !isManual) break;
    }

    const curJob = await getJob();
    const curCfg = await settings();
    const sourceLabel = formatSourceLabel();
    const sourceSuffix = sourceLabel ? `\nSession source: ${sourceLabel}.` : "";

    if (curJob.revision !== startRevision) {
      return { ok: true, imported, checkpointed: true, full_account: false, job: curJob, ...(sourceLabel ? { session_source: sourceLabel } : {}) };
    }

    curJob.attempt = 0;
    curJob.nextAttemptAt = null;

    if (completed) {
      curCfg.automatic = false;
      await chrome.storage.local.set({ settings: curCfg });
      await chrome.alarms.clear("chat-recall-sync");
      curJob.enabled = false;
      curJob.state = "complete";
      await saveJob(curJob);
    } else if (curCfg.automatic) {
      const nextBatchAt = Date.now() + 60000;
      curJob.enabled = true;
      curJob.state = "waiting";
      curJob.nextAttemptAt = nextBatchAt;
      await saveJob(curJob);
      await chrome.alarms.create("chat-recall-sync", { when: nextBatchAt });
    } else {
      curJob.enabled = false;
      curJob.state = "paused";
      await saveJob(curJob);
    }

    const updated = new Date().toISOString();
    const nextStepText = completed
      ? "Ordinary-list import finished."
      : (curCfg.automatic ? "Scheduled next batch." : "Run again or manually resume to continue.");
    await setStatus(completed ? "Ordinary-list scan caught up" : "Backfill checkpoint saved", `${imported} changed conversations imported this run. ${nextStepText}${sourceSuffix}\nComplete account coverage, project/archive discovery, and native-app sync latency are unverified.`, { lastSuccess: updated, accountId: identity ? identity.accountId : undefined, ...(sourceLabel ? { session_source: sourceLabel } : {}) });
    return { ok: true, imported, checkpointed: true, full_account: false, job: curJob, ...(sourceLabel ? { session_source: sourceLabel } : {}) };
  } catch (error) {
    if (error.session_source) observedSources.add(error.session_source);
    const sourceLabel = formatSourceLabel();
    const sourceSuffix = sourceLabel ? `\nSession source: ${sourceLabel}.` : "";
    const extra = { error: true };
    if (error.status !== undefined) extra.status = error.status;
    if (error.stage !== undefined) extra.stage = error.stage;
    if (error.challenge !== undefined) extra.challenge = error.challenge;
    if (sourceLabel) extra.session_source = sourceLabel;

    let curJob = await getJob();
    const curCfg = await settings();

    if (!(error.status === 429 && !error.challenge)) {
      if (curJob.revision !== startRevision) {
        return { ok: false, error: error.message, job: curJob, ...(sourceLabel ? { session_source: sourceLabel } : {}) };
      }
      curJob.state = "attention";
      curJob.enabled = false;
      curCfg.automatic = false;
      await chrome.storage.local.set({ settings: curCfg });
      await chrome.alarms.clear("chat-recall-sync");
      await saveJob(curJob);

      const errorDetail = sourceLabel ? `${error.message}\nSession source: ${sourceLabel}.` : error.message;
      await setStatus("Import needs attention", errorDetail, extra);
      return {
        ok: false,
        error: error.message,
        ...(error.status !== undefined ? { status: error.status } : {}),
        ...(error.stage !== undefined ? { stage: error.stage } : {}),
        ...(error.challenge !== undefined ? { challenge: error.challenge } : {}),
        job: curJob,
        ...(sourceLabel ? { session_source: sourceLabel } : {})
      };
    }

    // Remote HTTP 429 handling (rate-limit backoff)
    let localImportError = null;
    if (curJob.revision === startRevision && error.partial && config && identity && key && checkpoint && versions && Array.isArray(error.conversations) && error.conversations.length > 0) {
      const configMatches = curCfg.endpoint === config.endpoint && curCfg.token === config.token && curCfg.workspaceId === config.workspaceId;
      const accountMatches = error.accountId === identity.accountId;
      if (configMatches && accountMatches) {
        try {
          const coverage = { adapter: "chatgpt-web-experimental", status: "partial", account_id: error.accountId, last_success: new Date().toISOString(), ordinary_list_offset: offset, project_coverage: "unverified", archive_coverage: "unverified", native_app_sync: "unverified", full_account: false, has_missing_conversations: false };
          const report = await importLocal(curCfg, error.conversations, error.accountId, coverage);
          if (Array.isArray(report.errors) && report.errors.length) {
            throw new Error(`Local import was partial (${report.errors.length} errors).`);
          }
          Object.assign(versions, error.importedVersions);
          await chrome.storage.local.set({ [key]: { offset: checkpoint.offset, cursor: checkpoint.cursor, versions, phase: checkpoint.phase, pages: checkpoint.pages, updated: new Date().toISOString() } });
          imported += error.conversations.length;
        } catch (importErr) {
          localImportError = importErr.message;
        }
      }
    }

    if (error.retry_after_unrepresentable) {
      const freshJob = await getJob();
      const freshCfg = await settings();
      freshJob.state = "attention";
      freshJob.blocked = true;
      freshJob.nextAttemptAt = null;
      freshJob.enabled = false;
      freshCfg.automatic = false;
      await chrome.storage.local.set({ settings: freshCfg });
      await chrome.alarms.clear("chat-recall-sync");
      await saveJob(freshJob);
      const detailMsg = localImportError ? `Local import failed during rate limit handling: ${localImportError}` : `ChatGPT rate limit specified an unrepresentable delay. Backfill paused.${sourceSuffix}`;
      await setStatus("Import needs attention", detailMsg, { ...extra, blocked: true });
      return { ok: false, error: localImportError || error.message, status: 429, stage: error.stage, job: freshJob, ...(sourceLabel ? { session_source: sourceLabel } : {}) };
    }

    // Merge server cooldown minimum/attempt into fresh current job
    const freshJob = await getJob();
    const freshCfg = await settings();

    const attemptCandidate = (freshJob.attempt || 0) + 1;
    const backoff = computeBackoff(attemptCandidate, error.retry_after_ms);

    freshJob.attempt = Math.max(freshJob.attempt || 0, attemptCandidate);
    const mergedDeadline = freshJob.nextAttemptAt ? Math.max(freshJob.nextAttemptAt, backoff.nextAttemptAt) : backoff.nextAttemptAt;
    freshJob.nextAttemptAt = mergedDeadline;

    if (localImportError) {
      freshJob.state = "attention";
      freshJob.enabled = false;
      freshCfg.automatic = false;
      await chrome.storage.local.set({ settings: freshCfg });
      await chrome.alarms.clear("chat-recall-sync");
      await saveJob(freshJob);
      await setStatus("Import needs attention", `Local import failed during rate limit handling: ${localImportError}${sourceSuffix}`, { ...extra, error: true });
      return {
        ok: false,
        error: localImportError,
        status: 429,
        stage: error.stage,
        job: freshJob,
        ...(sourceLabel ? { session_source: sourceLabel } : {})
      };
    }

    if (freshCfg.automatic && freshJob.enabled) {
      freshJob.state = "cooldown";
      await saveJob(freshJob);
      await chrome.alarms.create("chat-recall-sync", { when: mergedDeadline });
    } else {
      if (freshJob.state !== "attention" && freshJob.state !== "complete") {
        freshJob.state = "paused";
      }
      await chrome.alarms.clear("chat-recall-sync");
      await saveJob(freshJob);
    }

    const errorDetail = sourceLabel ? `${error.message}\nSession source: ${sourceLabel}.` : error.message;
    await setStatus("Import needs attention", errorDetail, extra);
    return {
      ok: false,
      error: error.message,
      status: 429,
      ...(error.stage !== undefined ? { stage: error.stage } : {}),
      job: freshJob,
      ...(sourceLabel ? { session_source: sourceLabel } : {})
    };
  } finally {
    running = false;
  }
}

async function handleMessage(message) {
  if (!message || typeof message.action !== "string") throw new Error("Unknown bridge request.");
  await ensureInitialized();
  if (message.action === "status") {
    const cfg = await settings();
    const job = await getJob();
    job.enabled = cfg.automatic === true;
    return { ...(await chrome.storage.local.get("status")), settings: cfg, running, job };
  }
  if (message.action === "save") {
    const previous = await settings();
    const config = { ...previous, endpoint: adapter.endpoint(message.settings.endpoint), token: String(message.settings.token || "").trim(), workspaceId: String(message.settings.workspaceId || "").trim() };
    if (!config.token) throw new Error("Paste the pairing token from your local Chat Recall app.");
    config.automatic = false;
    await chrome.storage.local.set({ settings: config });
    await chrome.alarms.clear("chat-recall-sync");
    const job = await getJob();
    job.revision = (job.revision || 0) + 1;
    job.enabled = false;
    job.state = "paused";
    await saveJob(job);
    await setStatus("Pairing saved", "Start an account import to test the endpoint, token, and extension-origin allowlist.");
    return { ok: true, job };
  }
  if (message.action === "automatic") {
    const isEnabling = message.enabled === true;
    const cfg = await settings();
    if (isEnabling && !cfg.token) throw new Error("Save pairing before enabling automatic checks.");
    const job = await getJob();
    if (isEnabling && job.blocked) {
      throw new Error("Cannot enable automatic sync while blocked due to unrepresentable rate limit delay. Manual resolution required.");
    }
    cfg.automatic = isEnabling;
    await chrome.storage.local.set({ settings: cfg });

    job.revision = (job.revision || 0) + 1;
    job.enabled = isEnabling;
    if (!isEnabling) {
      job.state = "paused";
      await chrome.alarms.clear("chat-recall-sync");
      await saveJob(job);
      return { ok: true, job };
    } else {
      const now = Date.now();
      if (job.nextAttemptAt && now < job.nextAttemptAt) {
        job.state = "cooldown";
        await saveJob(job);
        await chrome.alarms.create("chat-recall-sync", { when: job.nextAttemptAt });
      } else {
        job.state = "waiting";
        job.nextAttemptAt = null;
        await saveJob(job);
        await chrome.alarms.create("chat-recall-sync", { when: now });
      }
      return { ok: true, job };
    }
  }
  if (message.action === "sync") return syncHistory({ manual: true });
  if (message.action === "reset") {
    if (running) throw new Error("Wait for the current import to finish before restarting its scan.");
    const cfg = await settings();
    if (cfg.automatic) throw new Error("Disable automatic checks before resetting the scan.");

    const job = await getJob();
    if (job.blocked) {
      throw new Error("Cannot reset scan while blocked due to unrepresentable rate limit delay. Manual resolution required.");
    }
    const now = Date.now();
    if (job.nextAttemptAt && now < job.nextAttemptAt) {
      throw new Error(`Backoff cooldown active until ${new Date(job.nextAttemptAt).toISOString()}. Wait before restarting.`);
    }

    running = true;
    try {
      const identity = await readChatGPT(cfg, { operation: "identity" });
      await chrome.storage.local.remove(`checkpoint:${identity.accountId}`);
      await setStatus("Account scan restarted", "The next import starts at the first page. Existing local conversations are preserved.");
      return { ok: true, job };
    } finally {
      running = false;
    }
  }
  if (message.action === "file") {
    const config = await settings();
    if (!config.token) throw new Error("Save pairing first.");
    const report = await importLocal(config, message.payload, "file-import", { adapter: "browser-json-file", status: "file-import", full_account: false });
    await setStatus("File import finished", `${report.conversations || 0} conversations · ${report.messages || 0} messages · ${report.skipped || 0} skipped. Account namespace: file-import.${report.errors?.length ? ` ${report.errors.length} import errors require attention.` : ""}`);
    return { ok: !report.errors?.length, report };
  }
  throw new Error("Unknown bridge request.");
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id) return false;
  handleMessage(message).then(sendResponse).catch((error) => sendResponse({ ok: false, error: error.message }));
  return true;
});

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name !== "chat-recall-sync") return;
  await ensureInitialized();
  if (running) return;
  const cfg = await settings();
  if (!cfg.automatic) {
    await chrome.alarms.clear("chat-recall-sync");
    return;
  }
  if (running) return;
  const job = await getJob();
  const now = Date.now();
  if (job.nextAttemptAt && now < job.nextAttemptAt) {
    await chrome.alarms.create("chat-recall-sync", { when: job.nextAttemptAt });
    return;
  }
  if (running) return;
  await syncHistory({ manual: false });
});

chrome.runtime.onStartup.addListener(async () => {
  await ensureInitialized();
  if (!running) {
    await reconcileJobAndAlarms();
  }
});

ensureInitialized().catch(() => {});
