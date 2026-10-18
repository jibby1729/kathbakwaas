"""Compare near-limit inputs, cancellation, and cache release to stock inference."""
import json
import os
from pathlib import Path
import sys
from threading import Event, Timer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['KOSHUR_OPTIMIZE_EXECUTION'] = '0'
import torch
from accuracy_corpus import GROUPS
from app.engine import TranslationEngine, TranslationCancelled
from app.fast_execution import optimize_fast_execution
from app.gpu_memory import GPUMemoryBusy

texts = [GROUPS['reported'][0]]
for a, b in [(0, 4), (3, 15), (7, 10)]:
    text = (GROUPS['long_passages'][a] + ' ' + GROUPS['long_passages'][b]
            + ' Please tell me if there is anything else I should bring.')
    assert len(text) <= 500
    texts.append(text)
report = {'texts': texts, 'rows': []}
path = ROOT / 'artifacts/accuracy-memory-stress.json'
engine = TranslationEngine()
engine.load()
assert engine.state == 'ready', engine.error
original_generate = engine.model.generate
tokens = None

def generate(*args, **kwargs):
    global tokens
    result = original_generate(*args, **kwargs)
    tokens = result.cpu().tolist()
    return result

engine.model.generate = generate

def memory():
    free, total = torch.cuda.mem_get_info()
    return {'free_mib': free / 2**20, 'allocated_mib': torch.cuda.memory_allocated()/2**20,
            'reserved_mib': torch.cuda.memory_reserved()/2**20,
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'cuda_oom_count': torch.cuda.memory_stats().get('num_ooms', 0)}

references = {}

def run(phase, text, session=None):
    torch.cuda.reset_peak_memory_stats()
    result = engine.translate(text, Event(), session)
    row = {'phase': phase, 'text': text, 'tokens': tokens, **result, 'memory': memory()}
    if phase == 'stock':
        references[text] = row
    else:
        ref = references[text]
        row['identical'] = all(row[k] == ref[k] for k in ['tokens', 'draft', 'translation'])
    report['rows'].append(row)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(phase, len(text), row.get('identical'), row['timings'], row['memory'], flush=True)
    assert row.get('identical', True)

for text in texts:
    run('stock', text)
os.environ['KOSHUR_OPTIMIZE_EXECUTION'] = '1'
engine.fast_execution = optimize_fast_execution(engine.model)
assert engine.fast_execution is not None
for text in texts:
    run('native', text)
for index in range(8):
    engine.prefill(texts[1 + index % 3], f'stress-{index}')
assert len(engine.prompt_cache.entries) <= 4
report['memory_with_four_sessions'] = memory()
report['cached_sessions'] = len(engine.prompt_cache.entries)
run('four_sessions', texts[0])

cancelled = Event()
timer = Timer(.15, cancelled.set)
timer.start()
try:
    engine.translate(texts[-1], cancelled)
    raise AssertionError('Expected cancellation during long generation')
except TranslationCancelled:
    report['in_flight_cancelled'] = True
finally:
    timer.cancel()
run('after_cancel', texts[0])

# Aim for 1400 MiB free, crossing the admission threshold. Allocation rounding
# and other applications can change the actual free amount; record it below.
# This tests cache release, not exhausting CUDA memory.
torch.cuda.synchronize()
torch.cuda.empty_cache()
free, _ = torch.cuda.mem_get_info()
size = max(0, int(free - 1400 * 2**20))
pressure = torch.empty(size, dtype=torch.uint8, device='cuda')
pressure.fill_(0)
torch.cuda.synchronize()
report['pressure_memory'] = memory()
if report['pressure_memory']['free_mib'] < 1024:
    del pressure
    torch.cuda.empty_cache()
    raise RuntimeError('Pressure probe left too little headroom; released it without inference')
try:
    run('cache_release', texts[0])
    report['pressure_result'] = 'completed after reclaim'
except GPUMemoryBusy:
    report['pressure_result'] = 'safely deferred before inference'
finally:
    del pressure
    torch.cuda.empty_cache()
run('after_pressure', texts[0])
report['completed'] = True
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('STRESS_COMPLETE', report['pressure_result'], flush=True)
