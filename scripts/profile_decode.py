"""Isolated profiling of the unchanged NF4/four-beam decoder."""
import os
import sys
import json
from pathlib import Path
from threading import Event

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['KOSHUR_OPTIMIZE_EXECUTION'] = '0'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import torch
from app.engine import TranslationEngine

engine = TranslationEngine()
engine.load()
assert engine.state == 'ready', engine.error
source = "oh my god you are driving me crazy and you are so annoying and i can't stand you"
for i in range(2):
    result = engine.translate(source, Event())
    print(json.dumps({'baseline': i, **result}, ensure_ascii=True), flush=True)
encoded = engine.encode(source)
with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    engine.model.generate(**encoded, num_beams=4, length_penalty=.7, early_stopping=True, do_sample=False, max_new_tokens=8, use_cache=True)
    torch.cuda.synchronize()
table = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=50)
print(table, flush=True)
(ROOT/'artifacts'/'decode-profile.txt').write_text(table, encoding='utf-8')
prof.export_chrome_trace(str(ROOT/'artifacts'/'decode-trace.json'))
print('DONE', flush=True)
