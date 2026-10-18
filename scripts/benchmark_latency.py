"""Local latency comparison. Stops no services and changes no app settings.

Run while the app is stopped so only one translator occupies the GPU.
Results include outputs for review, not a translation-quality score.
"""
import json
import argparse
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".model-cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from app.engine import TranslationEngine, TRANSLATOR_REVISION, RESTORER_REVISION

SAMPLES = [
    "How are you?",
    "Have you had tea?",
    "The weather is very pleasant today.",
    "Please close the window.",
    "I will call you tomorrow after I get home from work.",
]
engine = TranslationEngine()
parser = argparse.ArgumentParser()
parser.add_argument("--include-greedy", action="store_true", help="Compare output changes; never changes app decoding")
args = parser.parse_args()
beam_counts = [4, 1] if args.include_greedy else [4]
engine.load()
if engine.state != "ready":
    raise RuntimeError(engine.error)


class FirstStep(StoppingCriteria):
    def __init__(self):
        self.first = None
        self.steps = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.steps += 1
        if self.first is None:
            self.first = perf_counter()
        return False


def run(text, beams):
    started = perf_counter()
    encoded = engine.tokenizer.apply_chat_template([
        {"role": "system", "content": "Translate the text below to Kashmiri."},
        {"role": "user", "content": text},
    ], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(engine.model.device)
    torch.cuda.synchronize()
    tokenized = perf_counter()
    progress = FirstStep()
    options = {"length_penalty": 0.7, "early_stopping": True} if beams > 1 else {}
    with torch.inference_mode():
        tokens = engine.model.generate(
            **encoded, num_beams=beams, do_sample=False, max_new_tokens=512,
            use_cache=True,
            eos_token_id=[engine.tokenizer.convert_tokens_to_ids("<end_of_turn>"), engine.tokenizer.eos_token_id],
            stopping_criteria=StoppingCriteriaList([progress]), **options,
        )
    torch.cuda.synchronize()
    generated = perf_counter()
    new_tokens = tokens[0, encoded["input_ids"].shape[1]:]
    draft = engine.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    restored = engine.restorer.restore([draft])[0]
    completed = perf_counter()
    assert engine.strip_marks(draft) == engine.strip_marks(restored)
    return {
        "english": text, "beams": beams, "draft": draft, "translation": restored,
        "tokens": len(new_tokens), "steps": progress.steps,
        "tokenize_ms": round((tokenized - started) * 1000, 2),
        "generate_ms": round((generated - tokenized) * 1000, 2),
        "first_step_ms": round(((progress.first or generated) - tokenized) * 1000, 2),
        "restore_ms": round((completed - generated) * 1000, 2),
        "total_ms": round((completed - started) * 1000, 2),
    }


print("Warming up decoding paths", flush=True)
for beams in beam_counts:
    run(SAMPLES[0], beams)

rows = []
for repeat in range(2):
    for text in SAMPLES:
        # Reverse the order on the second repeat to reduce warm-up/order bias.
        for beams in (beam_counts if repeat == 0 else list(reversed(beam_counts))):
            row = {**run(text, beams), "repeat": repeat}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)

baseline = [r for r in rows if r["beams"] == 4 and r["repeat"] == 1]
cpu_restored = [r["translation"] for r in baseline]
engine.restorer.model.to("cuda")
engine.restorer.device = torch.device("cuda")
engine.restorer.restore([baseline[0]["draft"]])
torch.cuda.synchronize()
gpu_restorer = []
for row, cpu_text in zip(baseline, cpu_restored):
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        started = perf_counter()
        gpu_text = engine.restorer.restore([row["draft"]])[0]
        torch.cuda.synchronize()
        times.append((perf_counter() - started) * 1000)
    gpu_restorer.append({
        "english": row["english"], "gpu_restore_ms": round(statistics.median(times), 2),
        "matches_cpu": gpu_text == cpu_text, "translation": gpu_text,
    })

summary = {}
for beams in beam_counts:
    subset = [r for r in rows if r["beams"] == beams]
    summary[str(beams)] = {
        key: round(statistics.median(r[key] for r in subset), 2)
        for key in ["total_ms", "generate_ms", "restore_ms", "first_step_ms"]
    }
report = {
    "gpu": torch.cuda.get_device_name(), "translator_revision": TRANSLATOR_REVISION,
    "restorer_revision": RESTORER_REVISION, "quantization": "NF4 + double quant, BF16 compute",
    "summary": summary, "runs": rows, "gpu_restorer": gpu_restorer,
}
path = ROOT / "artifacts" / "latency-benchmark.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("SUMMARY " + json.dumps(summary), flush=True)
print("GPU_RESTORER " + json.dumps(gpu_restorer, ensure_ascii=True), flush=True)
print(f"Saved {path}", flush=True)
