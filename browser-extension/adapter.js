"use strict";

// Pure utilities shared by the background bridge and offline adapter checks.
// No ChatGPT credentials or conversation content are kept in checkpoint state.
globalThis.ChatRecallAdapter = Object.freeze({
  endpoint(value) {
    const url = new URL(value);
    if (url.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(url.hostname) || url.username || url.password || url.search || url.hash || url.pathname !== "/") {
      throw new Error("Use an HTTP loopback endpoint such as http://127.0.0.1:8765, with no path or credentials.");
    }
    return url.origin;
  },
  version(item) {
    const value = item.update_time ?? item.create_time;
    if (typeof value === "number" && Number.isFinite(value)) return String(Math.round(value * (value < 100000000000 ? 1000 : 1)));
    if (typeof value === "string") { const milliseconds = Date.parse(value); if (Number.isFinite(milliseconds)) return String(milliseconds); }
    return null;
  },
  page(payload, offset, limit) {
    if (!payload || !Array.isArray(payload.items)) throw new Error("The ChatGPT conversation-list format changed. Nothing was marked fully synced.");
    if (payload.items.some((item) => !item || typeof item.id !== "string" || !item.id)) throw new Error("The conversation list contains a record without an ID. The scan checkpoint was preserved.");
    const total = typeof payload.total === "number" && Number.isFinite(payload.total) ? payload.total : null;
    const cursor = typeof payload.cursor === "string" && payload.cursor ? payload.cursor : null;
    const exhausted = payload.items.length === 0 || payload.has_more === false || total !== null && offset + payload.items.length >= total || payload.items.length < limit && payload.has_more !== true && !cursor;
    return { items: payload.items, total, cursor, exhausted, missing: payload.has_missing_conversations === true };
  },
  checkpoint(value) {
    return value && typeof value === "object" && Number.isInteger(value.offset) && value.offset >= 0 ? value : { offset: 0, cursor: null, versions: {}, phase: "backfill", pages: 0 };
  }
});
