"""Compare prefix reuse with fresh four-beam runs on the cached local models."""
import json
import os
from pathlib import Path
import sys
from threading import Event
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".model-cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from app.engine import TranslationEngine

engine = TranslationEngine()
engine.load()
if engine.state != "ready":
    raise RuntimeError(engine.error)
engine.translate("Hello.", Event())
original = "oh my god you are driving me crazy and you are so annoying and i can't stand you"
cases = [
    ("append words", ["How", "How are", "How are you?"], "How are you?"),
    ("long input", ["oh my god", "oh my god you are driving me crazy", original], original),
    ("append two words", [original, original + " right now"], original + " right now"),
    ("delete words", [original, "oh my god you are driving me crazy"], "oh my god you are driving me crazy"),
    ("edit earlier words", [original, original.replace("annoying", "kind")], original.replace("annoying", "kind")),
    ("replace source", [original, "The weather is very pleasant today."], "The weather is very pleasant today."),
    ("partial word", ["Please close the wind", "Please close the window."], "Please close the window."),
]
report = []
for label, updates, text in cases:
    engine.prompt_cache.discard("probe")
    baseline = engine.translate(text, Event())
    prefixes = []
    for update in updates:
        prefixes.append(engine.prefill(update, "probe"))
    started = perf_counter()
    cached = engine.translate(text, Event(), session="probe")
    row = {"case": label, "english": text, "baseline": baseline, "cached": cached,
           "cache_request_ms": round((perf_counter() - started) * 1000, 2),
           "prefill_updates": prefixes,
           "identical_draft": baseline["draft"] == cached["draft"],
           "identical_translation": baseline["translation"] == cached["translation"]}
    report.append(row)
    print(json.dumps(row, ensure_ascii=True), flush=True)
    (ROOT / "artifacts" / "prefix-benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("ALL_IDENTICAL " + str(all(r["identical_translation"] and r["identical_draft"] for r in report)), flush=True)
