import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const nodes = new Map();
function node(id) {
  if (!nodes.has(id)) nodes.set(id, {disabled: false, textContent: '', attributes: {},
    handlers: {}, setAttribute(k, v) { this.attributes[k] = v; },
    addEventListener(k, f) { this.handlers[k] = f; }});
  return nodes.get(id);
}
const requests = [], audios = [], revoked = [], toasts = [];
let nextResponse;
const context = vm.createContext({
  document: {getElementById: node}, AbortController, setTimeout, clearTimeout,
  addEventListener() {}, showToast: m => toasts.push(m),
  URL: {createObjectURL: () => 'blob:test', revokeObjectURL: u => revoked.push(u)},
  Audio: class {
    constructor() { this.handlers = {}; audios.push(this); }
    addEventListener(k, f) { this.handlers[k] = f; }
    async play() { this.played = true; }
    pause() { this.paused = true; }
    removeAttribute() {} load() {}
  },
  async fetch(url, options) {
    requests.push({url, ...JSON.parse(options.body)});
    if (nextResponse) return await nextResponse;
    return {ok: true, blob: async () => ({})};
  },
});
vm.runInContext(fs.readFileSync('web/speech.js', 'utf8'), context);
const flush = () => new Promise(setImmediate);
const click = v => node(`voice-${v}`).handlers.click();
assert.equal(node('voice-male').disabled, true);
context.speechControls.setText('چون ناو کیا چھُ؟');
await click('male');
assert.equal(requests[0].voice, 'male');
assert.equal(audios[0].played, true);
assert.equal(node('voice-male-label').textContent, 'Stop');
await click('male');
assert.equal(audios[0].paused, true);
await click('male');
assert.equal(requests.length, 1, 'Replay uses current-translation cache');
await click('female');
assert.equal(audios[1].paused, true, 'Switching voices stops old audio');
assert.equal(requests.at(-1).voice, 'female');
context.speechControls.setText('');
assert.equal(audios.at(-1).paused, true);
assert.equal(node('voice-male').disabled, true);
context.speechControls.setText('کٲشُر');
let resolve;
nextResponse = new Promise(r => resolve = r);
const pending = click('male');
await flush();
assert.equal(node('voice-male-label').textContent, 'Loading…');
assert.equal(node('voice-male').disabled, true);
assert.equal(node('voice-female').disabled, true);
const requestCount = requests.length;
await click('male');
await click('female');
assert.equal(requests.length, requestCount, 'Repeated clicks while loading must be ignored');
resolve({ok: true, blob: async () => ({})});
await pending;
assert.equal(audios.at(-1).played, true, 'Loading must finish with automatic playback despite repeated clicks');
assert.equal(node('voice-male-label').textContent, 'Stop');
assert.equal(node('voice-male').disabled, false);
assert.equal(node('voice-female').disabled, false);
context.speechControls.setText('نَو متن');
nextResponse = new Promise(r => resolve = r);
const stalePending = click('male');
await flush();
context.speechControls.setText('another translation');
const count = audios.length;
resolve({ok: true, blob: async () => ({})});
await stalePending;
assert.equal(audios.length, count, 'Late responses cannot play stale translations');
nextResponse = Promise.resolve({ok: false, json: async () => ({detail: 'Speech is busy.'})});
await click('female');
assert.equal(toasts.at(-1), 'Speech is busy.');
assert.equal(node('voice-female-label').textContent, 'Female voice');
assert.equal(node('voice-female').disabled, false, 'Errors must allow retry');
assert.ok(revoked.length >= 3);
console.log('Speech UI passed: voices, replay, stop, edits, stale response, errors, URL cleanup.');
