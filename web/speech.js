"use strict";

(() => {
  const voices = ["male", "female"];
  const buttons = Object.fromEntries(voices.map(v => [v, document.getElementById(`voice-${v}`)]));
  const labels = Object.fromEntries(voices.map(v => [v, document.getElementById(`voice-${v}-label`)]));
  const status = document.getElementById("speech-status");
  let text = "";
  let generation = 0;
  let activeVoice = null;
  let loading = false;
  let request = null;
  let audio = null;
  let audioUrl = null;
  const cache = new Map(); // At most two voices for the current translation.

  function updateButtons() {
    for (const voice of voices) {
      const active = activeVoice === voice;
      buttons[voice].disabled = !text || loading;
      buttons[voice].setAttribute("aria-pressed", String(active));
      buttons[voice].setAttribute("aria-label", active
        ? `${loading ? "Loading" : "Stop"} ${voice} voice`
        : `Listen in a ${voice} voice`);
      labels[voice].textContent = active
        ? loading ? "Loading…" : "Stop"
        : `${voice === "male" ? "Male" : "Female"} voice`;
    }
  }

  function stop() {
    ++generation;
    request?.abort();
    request = null;
    if (audio) {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
      audio = null;
    }
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = null;
    activeVoice = null;
    loading = false;
    status.textContent = "";
    updateButtons();
  }

  async function play(voice) {
    if (!text || loading) return;
    if (activeVoice === voice) { stop(); return; }
    stop();
    const mine = generation;
    activeVoice = voice;
    loading = true;
    updateButtons();
    status.textContent = `Preparing ${voice} voice…`;
    const controller = new AbortController();
    request = controller;
    const timeout = setTimeout(() => controller.abort("timeout"), 210000);
    try {
      let blob = cache.get(voice);
      if (!blob) {
        const response = await fetch("/api/speech", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, voice }), signal: controller.signal,
        });
        if (!response.ok) {
          const data = await response.json();
          throw new Error(typeof data.detail === "string" ? data.detail : "Audio could not be made. Try again.");
        }
        blob = await response.blob();
        if (mine !== generation) return;
        cache.set(voice, blob);
      }
      if (mine !== generation) return;
      audioUrl = URL.createObjectURL(blob);
      audio = new Audio(audioUrl);
      audio.addEventListener("ended", () => { if (mine === generation) stop(); });
      audio.addEventListener("error", () => {
        if (mine !== generation) return;
        stop();
        showToast("Audio could not play. Please try again.");
      });
      await audio.play();
      if (mine !== generation) return;
      loading = false;
      updateButtons();
      status.textContent = `Playing ${voice} voice. Click again to stop.`;
    } catch (error) {
      if (mine !== generation) return;
      stop();
      const message = controller.signal.reason === "timeout" ? "Audio took too long. Try a shorter sentence."
        : error.name === "NotAllowedError" ? "Tap the voice button again to play."
        : error instanceof TypeError ? "Couldn’t reach the audio service. Try again." : error.message;
      status.textContent = message;
      showToast(message);
    } finally {
      clearTimeout(timeout);
      if (request === controller) request = null;
    }
  }

  globalThis.speechControls = {
    setText(value) {
      if (text === value) return;
      text = value;
      stop();
      cache.clear();
    },
  };
  for (const voice of voices) buttons[voice].addEventListener("click", () => play(voice));
  globalThis.addEventListener("pagehide", stop);
  updateButtons();
})();
