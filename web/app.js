"use strict";

const $ = (id) => document.getElementById(id);
const input = $("english");
const output = $("translation");
const empty = $("empty-output");
const status = $("translation-status");
const copyButton = $("copy");
const connection = $("connection");
const DEBOUNCE_MS = 250;
const PREFILL_DELAY_MS = 120;
const OVERLOAD_MESSAGE = "Server overloaded. Too many users or limited server memory. Please try again shortly.";
const UNAVAILABLE_MESSAGE = "Server unavailable or overloaded. Please try again shortly.";
// A curated example supplied by the site owner, not a model-generated result.
const savedTranslations = new Map([
  ["kashmir belongs to kashmiris", "کٔشیٖر چھِ کٲشِرؠن ہٕنٛز۔"],
]);
function savedTranslation(text) {
  return savedTranslations.get(text.trim().toLowerCase().replace(/[.!?]$/, ""));
}
const sessionId = crypto.randomUUID();
let timer;
let prefillTimer;
let prefillRequest;
let activeRequest;
let revision = 0;
let result = null;
let ready = false;
let composing = false;
let lastRequested = "";
let toastTimer;

function announce(message, error = false) {
  status.textContent = message;
  status.dataset.error = String(error);
}

function setBusy(busy) {
  $("output-panel").setAttribute("aria-busy", String(busy));
}

function resetOutput() {
  globalThis.speechControls?.setText("");
  result = null;
  output.textContent = "";
  output.hidden = true;
  empty.hidden = false;
  empty.textContent = "Translation";
  $("translation-error").hidden = true;
  copyButton.disabled = true;
  $("copy-label").textContent = "Copy";
  setBusy(false);
}

function renderResult() {
  if (!result) return;
  // Keep pronunciation marks even when the display toggle hides them.
  globalThis.speechControls?.setText(result.translation);
  output.textContent = $("reading-marks").checked ? result.translation : result.draft;
  output.hidden = false;
  empty.hidden = true;
  $("translation-error").hidden = true;
  copyButton.disabled = false;
}

function showError(message) {
  resetOutput();
  empty.hidden = true;
  $("translation-error").hidden = false;
  $("error-message").textContent = message;
  announce("Translation failed.", true);
}

function showToast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
}

async function translateNow() {
  clearTimeout(timer);
  clearTimeout(prefillTimer);
  const text = input.value.trim();
  if (!text || composing) return;
  const saved = savedTranslation(text);
  if (saved) {
    ++revision;
    activeRequest?.abort();
    prefillRequest?.abort();
    activeRequest = null;
    lastRequested = text;
    result = { translation: saved, draft: saved.replace(/\p{M}/gu, "") };
    renderResult();
    setBusy(false);
    announce("Saved translation ready.");
    return;
  }
  if (!ready) {
    announce("Waiting for the translator…");
    return;
  }
  const myRevision = ++revision;
  activeRequest?.abort();
  const controller = new AbortController();
  activeRequest = controller;
  lastRequested = text;
  resetOutput();
  setBusy(true);
  empty.textContent = "Translating…";
  announce("Translating…");
  // A local model may need time, especially for a paragraph.
  const timeout = setTimeout(() => controller.abort("timeout"), 180000);
  const slowNotice = setTimeout(() => {
    if (myRevision !== revision || controller.signal.aborted) return;
    empty.textContent = "The server is busy. Still trying…";
    announce("The server is busy. Still trying…");
  }, 15000);
  try {
    const response = await fetch("/api/translate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, session_id: sessionId }), signal: controller.signal,
    });
    if (myRevision !== revision) return;
    // Capacity and proxy failures may return HTML rather than JSON.
    if (response.status === 429 || response.status === 503) throw new Error(OVERLOAD_MESSAGE);
    if (response.status >= 500) throw new Error(UNAVAILABLE_MESSAGE);
    const data = await response.json();
    if (myRevision !== revision) return;
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "Please try a shorter English sentence.";
      throw new Error(detail);
    }
    if (typeof data.translation !== "string" || typeof data.draft !== "string") {
      throw new Error("The translator sent an incomplete response. Please try again.");
    }
    result = data;
    renderResult();
    announce("Translation ready.");
  } catch (error) {
    if (myRevision !== revision) return;
    if (controller.signal.aborted && controller.signal.reason !== "timeout") return;
    showError(controller.signal.reason === "timeout"
      ? UNAVAILABLE_MESSAGE
      : error instanceof TypeError
        ? "We couldn’t reach the translator. Check the connection and try again."
        : error.message);
  } finally {
    clearTimeout(timeout);
    clearTimeout(slowNotice);
    if (myRevision === revision) {
      setBusy(false);
      activeRequest = null;
    }
  }
}

async function prefillNow(expectedRevision) {
  if (expectedRevision !== revision || !ready || composing || activeRequest) return;
  const text = input.value.trim();
  if (!text || savedTranslation(text)) return;
  prefillRequest?.abort();
  const controller = new AbortController();
  prefillRequest = controller;
  try {
    await fetch("/api/prefill", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, session_id: sessionId }), signal: controller.signal,
    });
  } catch { /* Speculative prefill is optional; translation can always run normally. */ }
  finally { if (prefillRequest === controller) prefillRequest = null; }
}

function onInput() {
  clearTimeout(timer);
  clearTimeout(prefillTimer);
  ++revision;
  prefillRequest?.abort();
  activeRequest?.abort();
  activeRequest = null;
  lastRequested = "";
  const count = input.value.length;
  $("character-count").textContent = `${count} / 500`;
  $("character-count").classList.toggle("at-limit", input.value.length >= 500);
  $("clear").disabled = input.value.length === 0;
  resetOutput();
  if (!input.value.trim()) {
    announce("");
    return;
  }
  if (!composing && savedTranslation(input.value)) {
    translateNow();
    return;
  }
  announce(ready ? "" : "Waiting for the translator…");
  if (!composing) {
    const expectedRevision = revision;
    // Start input processing at word boundaries, or after a brief gap between keys.
    prefillTimer = setTimeout(() => prefillNow(expectedRevision), /\s$/.test(input.value) ? 0 : PREFILL_DELAY_MS);
    timer = setTimeout(translateNow, DEBOUNCE_MS);
  }
}

input.addEventListener("input", onInput);
input.addEventListener("compositionstart", () => {
  composing = true;
  clearTimeout(timer);
  clearTimeout(prefillTimer);
});
input.addEventListener("compositionend", () => { composing = false; onInput(); });
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    translateNow();
  }
});
$("clear").addEventListener("click", () => { input.value = ""; onInput(); input.focus(); });
$("retry").addEventListener("click", async () => {
  await checkHealth();
  if (!activeRequest) translateNow();
});
$("reading-marks").addEventListener("change", renderResult);
document.querySelectorAll("[data-phrase]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.phrase;
    onInput();
    translateNow();
    input.focus({ preventScroll: true });
    if (matchMedia("(max-width: 650px)").matches) {
      $("english").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "center" });
    }
  });
});

$("paste").addEventListener("click", async () => {
  try {
    const text = await navigator.clipboard.readText();
    if (!text) { showToast("Your clipboard is empty."); return; }
    input.value = text.slice(0, 500);
    onInput();
    input.focus();
    if (text.length > 500) showToast("Pasted the first 500 characters.");
  } catch {
    input.focus();
    showToast("Click the text box, then press Ctrl+V (or ⌘V) to paste.");
  }
});

copyButton.addEventListener("click", async () => {
  if (!result) return;
  try {
    await navigator.clipboard.writeText(output.textContent);
    $("copy-label").textContent = "Copied!";
    showToast("Copied.");
    setTimeout(() => { $("copy-label").textContent = "Copy"; }, 2500);
  } catch {
    const selection = getSelection();
    const range = document.createRange();
    range.selectNodeContents(output);
    selection.removeAllRanges();
    selection.addRange(range);
    showToast("Text selected. Press Ctrl+C (or ⌘C) to copy.");
  }
});

function setLargeText(enabled) {
  document.body.classList.toggle("large-text", enabled);
  $("text-size").setAttribute("aria-pressed", String(enabled));
  try { localStorage.setItem("kath-bakwaas-large-text", String(enabled)); } catch { /* Optional preference only. */ }
}
$("text-size").addEventListener("click", () => setLargeText(!document.body.classList.contains("large-text")));
try { setLargeText(localStorage.getItem("kath-bakwaas-large-text") === "true"); } catch { /* Storage can be disabled. */ }

$("about-open").addEventListener("click", () => $("about-dialog").showModal());
$("about-close").addEventListener("click", () => $("about-dialog").close());
$("about-dialog").addEventListener("click", (event) => {
  if (event.target !== $("about-dialog")) return;
  const bounds = event.target.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) event.target.close();
});

let healthInFlight = false;
async function checkHealth() {
  if (healthInFlight) return;
  healthInFlight = true;
  const wasReady = ready;
  try {
    const response = await fetch("/api/health", { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error("Server unavailable");
    const data = await response.json();
    ready = data.state === "ready";
    connection.dataset.state = data.state;
    $("connection-text").textContent = ready ? "Ready" : ({
      starting: "Starting…", downloading: "Downloading models…", loading: "Loading models…",
      error: "Translator unavailable", offline: "Preview mode",
    }[data.state] || "Loading…");
    connection.title = data.message || "";
    if (!ready && !result) {
      announce(data.message || "Getting the translator ready…", data.state === "error");
    }
    if (ready && !wasReady) {
      if (input.value.trim() && lastRequested !== input.value.trim()) {
        const expectedRevision = revision;
        clearTimeout(prefillTimer);
        clearTimeout(timer);
        prefillTimer = setTimeout(() => prefillNow(expectedRevision), 0);
        timer = setTimeout(translateNow, DEBOUNCE_MS);
      }
      else if (!input.value.trim()) announce("");
    }
  } catch {
    ready = false;
    connection.dataset.state = "offline";
    $("connection-text").textContent = "Connection lost";
    if (!result) announce("We couldn’t reach the translator. Reconnecting…", true);
  } finally {
    healthInFlight = false;
  }
}
checkHealth();
setInterval(() => { if (!document.hidden) checkHealth(); }, 5000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) checkHealth(); });
