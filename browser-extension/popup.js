"use strict";
const $ = (id) => document.getElementById(id);
let timer;
let actionError = false;

function formatJob(job) {
  if (!job || !job.state) return "";
  const stateLabels = {
    waiting: "Waiting",
    running: "Running",
    cooldown: "Cooldown",
    paused: "Paused",
    attention: "Needs attention",
    complete: "Import complete",
  };
  const label = stateLabels[job.state] || job.state;
  const parts = [];
  if (typeof job.attempt === "number" && job.attempt > 0) {
    parts.push(`${label} (attempt ${job.attempt})`);
  } else {
    parts.push(label);
  }
  if (typeof job.nextAttemptAt === "number" && Number.isFinite(job.nextAttemptAt) && job.nextAttemptAt > 0) {
    parts.push(`Next scheduled attempt: ${new Date(job.nextAttemptAt).toLocaleString()}`);
  }
  return parts.join(" · ");
}

async function send(action, values = {}) {
  const response = await chrome.runtime.sendMessage({ action, ...values });
  if (!response) throw new Error("The background bridge did not respond.");
  if (action !== "sync" && response.ok === false && response.error) {
    throw new Error(response.error);
  }
  return response;
}

function error(message) {
  actionError = true;
  $("status-title").textContent = "Needs attention";
  const raw = String(message || "The background bridge did not respond.");
  $("status-detail").textContent = raw.length > 240 ? raw.slice(0, 240) : raw;
  $("status-detail").classList.add("error");
}

async function refresh(fill = false) {
  try {
    const data = await send("status");
    if (fill && data.settings) {
      if (data.settings.endpoint !== undefined) $("endpoint").value = data.settings.endpoint;
      if (data.settings.token !== undefined) $("token").value = data.settings.token;
      if (data.settings.workspaceId !== undefined) $("workspace-id").value = data.settings.workspaceId || "";
    }
    if (data.settings && typeof data.settings.automatic === "boolean") {
      $("automatic").checked = data.settings.automatic;
    }
    const hasFutureDeadline = typeof data.job?.nextAttemptAt === "number" &&
      Number.isFinite(data.job.nextAttemptAt) &&
      data.job.nextAttemptAt > Date.now();
    $("sync").disabled = Boolean(data.running || data.job?.state === "running" || hasFutureDeadline);

    if (actionError) return;
    $("status-title").textContent = data.status?.title || "Not paired";
    $("status-detail").textContent = data.status?.detail || "Save your endpoint and pairing token to begin.";
    $("status-detail").classList.toggle("error", data.status?.error === true);
    const jobStatusEl = $("job-status");
    if (jobStatusEl) jobStatusEl.textContent = formatJob(data.job);
    $("last-sync").textContent = data.status?.lastSuccess ? `Last successful run: ${new Date(data.status.lastSuccess).toLocaleString()}` : "";
  } catch (failure) {
    error(failure.message);
  }
}

$("extension-origin").textContent = `chrome-extension://${chrome.runtime.id}`;
$("settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  actionError = false;
  try {
    await send("save", { settings: { endpoint: $("endpoint").value, token: $("token").value, workspaceId: $("workspace-id").value } });
    await refresh(true);
  } catch (failure) {
    error(failure.message);
  }
});
$("sync").addEventListener("click", async () => {
  actionError = false;
  $("sync").disabled = true;
  try {
    await send("sync");
  } catch (failure) {
    error(failure.message);
  } finally {
    await refresh();
  }
});
$("reset").addEventListener("click", async () => {
  actionError = false;
  try {
    await send("reset");
    await refresh();
  } catch (failure) {
    error(failure.message);
  }
});
$("automatic").addEventListener("change", async () => {
  actionError = false;
  try {
    await send("automatic", { enabled: $("automatic").checked });
  } catch (failure) {
    $("automatic").checked = !$("automatic").checked;
    error(failure.message);
  } finally {
    await refresh();
  }
});
$("file").addEventListener("change", async () => {
  const file = $("file").files[0];
  if (!file) return;
  actionError = false;
  try {
    if (file.size > 64 * 1024 * 1024) throw new Error("The file exceeds 64 MB. Import smaller conversation batches using the local app.");
    const payload = JSON.parse(await file.text());
    await send("file", { payload });
    await refresh();
  } catch (failure) {
    error(failure.message);
  }
});
refresh(true);
timer = setInterval(() => refresh(), 1500);
window.addEventListener("unload", () => clearInterval(timer));
