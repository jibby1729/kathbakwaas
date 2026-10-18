import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

// Exercise the real browser script with a deterministic clock and HTTP stand-ins.
const nodes = new Map();
function node(id) {
  if (!nodes.has(id)) nodes.set(id, {
    value: "", textContent: "", hidden: false, checked: true, handlers: {}, dataset: {},
    classList: { toggle() {}, contains() { return false; } },
    addEventListener(name, callback) { this.handlers[name] = callback; },
    setAttribute() {}, focus() {},
  });
  return nodes.get(id);
}
let now = 0;
let nextTimer = 0;
const timers = new Map();
const requests = [];
const context = vm.createContext({
  document: { getElementById: node, querySelectorAll: () => [], hidden: false,
    addEventListener() {}, body: { classList: node("body").classList } },
  crypto: { randomUUID: () => "typing-test-page" },
  localStorage: { getItem: () => null, setItem() {} },
  AbortController, AbortSignal,
  setInterval() {},
  setTimeout(callback, delay) { const id = ++nextTimer; timers.set(id, { at: now + delay, callback }); return id; },
  clearTimeout(id) { timers.delete(id); },
  async fetch(url, options) {
    requests.push({ url, body: options?.body ? JSON.parse(options.body) : null });
    return { ok: true, json: async () => url === "/api/health" ? { state: "ready" }
      : url === "/api/prefill" ? { prefilled: true }
      : { translation: "کٲشُر", draft: "کاشر" } };
  },
});
vm.runInContext(readFileSync(new URL("../web/app.js", import.meta.url), "utf8"), context);
const flush = () => new Promise(setImmediate);
async function tick(delta) {
  const until = now + delta;
  for (;;) {
    const entry = [...timers].filter(([, t]) => t.at <= until).sort((a, b) => a[1].at - b[1].at)[0];
    if (!entry) break;
    const [id, timer] = entry;
    now = timer.at;
    timers.delete(id);
    timer.callback();
    await flush();
  }
  now = until;
}
function type(text) { node("english").value = text; node("english").handlers.input(); }
await flush();
type("How are");
await tick(119);
assert.equal(requests.filter(r => r.url === "/api/prefill").length, 0);
await tick(1);
assert.equal(requests.at(-1).url, "/api/prefill");
assert.equal(requests.at(-1).body.text, "How are");
await tick(129);
assert.equal(requests.filter(r => r.url === "/api/translate").length, 0);
await tick(1);
assert.equal(requests.at(-1).url, "/api/translate");
assert.equal(requests.at(-1).body.session_id, "typing-test-page");
assert.equal(node("translation").textContent, "کٲشُر");

type("How are you ");
await tick(0);
assert.equal(requests.at(-1).url, "/api/prefill");
assert.equal(requests.at(-1).body.text, "How are you");
type("");
const beforeClear = requests.length;
await tick(500);
assert.equal(requests.length, beforeClear, "Clearing must cancel pending work");

node("english").handlers.compositionstart();
type("composing");
const beforeComposition = requests.length;
await tick(500);
assert.equal(requests.length, beforeComposition, "Do not prefill incomplete IME input");
node("english").handlers.compositionend();
await tick(120);
assert.equal(requests.at(-1).url, "/api/prefill");
assert.equal(requests.at(-1).body.text, "composing");
type("Kashmir belongs to Kashmiris.");
const beforeSaved = requests.length;
assert.equal(node("translation").textContent, "کٔشیٖر چھِ کٲشِرؠن ہٕنٛز۔");
await tick(500);
assert.equal(requests.length, beforeSaved, "Saved phrase must not prefill or run inference");
node("reading-marks").checked = false;
node("reading-marks").handlers.change();
assert.equal(node("translation").textContent, "کشیر چھ کٲشرؠن ہنز۔");
type("Kashmir belongs to Kashmiris and others.");
await tick(250);
assert.equal(requests.at(-1).body.text, "Kashmir belongs to Kashmiris and others.", "Edited sentences must use the model");
console.log("Typing checks passed: early prefill, word boundaries, page session, debounce, clear, IME, saved phrase and diacritics.");
