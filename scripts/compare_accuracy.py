"""Record full outputs from a fresh model process and compare to stock references.

Run stock first, then native, then native --typing. No completed-answer cache.
Every case is saved immediately; failures are retained rather than skipped.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from accuracy_corpus import CASES

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['stock', 'native', 'conversion'], required=True)
parser.add_argument('--typing', action='store_true')
parser.add_argument('--limit', type=int)
parser.add_argument('--report')
parser.add_argument('--fail-fast', action='store_true')
parser.add_argument('--corpus', help='JSON list of id/group/text cases, relative to project root')
parser.add_argument('--reference', default='artifacts/accuracy-stock.json')
args = parser.parse_args()
if args.corpus:
    CASES = json.loads((ROOT / args.corpus).read_text(encoding='utf-8'))
assert len({c['id'] for c in CASES}) == len(CASES)
assert all(0 < len(c['text']) <= 500 for c in CASES)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['KOSHUR_OPTIMIZE_EXECUTION'] = '0' if args.backend == 'stock' else '1'
os.environ['KOSHUR_NATIVE_DECODE'] = '0' if args.backend == 'conversion' else '1'
import torch
import transformers
import bitsandbytes
logging.basicConfig(level=logging.INFO)
from app.engine import TranslationEngine, TRANSLATOR_REVISION, RESTORER_REVISION

report = ROOT / 'artifacts' / (args.report or f'accuracy-{args.backend}{"-typing" if args.typing else ""}.json')
reference_path = ROOT / args.reference
references = {}
if report != reference_path and reference_path.exists():
    references = {r['id']: r for r in json.loads(reference_path.read_text(encoding='utf-8'))['rows']}
    assert all(c['id'] in references and references[c['id']]['text'] == c['text']
               for c in CASES[:args.limit]), 'Incomplete or mismatched baseline'
elif args.backend != 'stock':
    raise RuntimeError(f'Missing baseline: {reference_path}')
rows = []
document = {'backend': args.backend, 'typing': args.typing, 'rows': rows,
            'corpus_sha256': hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest(),
            'translator_revision': TRANSLATOR_REVISION, 'restorer_revision': RESTORER_REVISION,
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'bitsandbytes': bitsandbytes.__version__, 'completed': False}
document['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / 'app').rglob('*.py'))}
document['reference'] = str(reference_path)

def save():
    report.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')

def memory():
    free, total = torch.cuda.mem_get_info()
    return {'free_mib': round(free / 2**20, 1), 'total_mib': round(total / 2**20, 1),
            'allocated_mib': round(torch.cuda.memory_allocated() / 2**20, 1),
            'reserved_mib': round(torch.cuda.memory_reserved() / 2**20, 1),
            'peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1)}

engine = TranslationEngine()
start = time.perf_counter()
engine.load()
assert engine.state == 'ready', engine.error
document.update(load_seconds=time.perf_counter()-start, optimization=engine.execution_optimization,
                ready_memory=memory(), generation_config=engine.model.generation_config.to_dict(),
                generation_options=engine.generation_options())
if args.backend == 'native':
    assert engine.execution_optimization['backend'] == 'native_nf4'
original_generate = engine.model.generate
tokens = None

def generate(*a, **kw):
    global tokens
    result = original_generate(*a, **kw)
    tokens = result.cpu().tolist()
    return result

engine.model.generate = generate
print('READY', json.dumps({k: document[k] for k in ['load_seconds', 'optimization', 'ready_memory']}), flush=True)
engine.translate('Hello.', Event())
for case in CASES[:args.limit]:
    tokens = None
    row = dict(case)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        session = None
        if args.typing:
            session = 'broad-accuracy'
            engine.prompt_cache.discard(session)
            words = case['text'].split(' ')
            # Prefix additions, an edit/deletion, then exact final text.
            middle = max(1, len(words)//2)
            versions = [' '.join(words[:middle]), ' '.join(words[:middle]) + ' yesterday',
                        ' '.join(words[:max(1,middle-1)]), case['text']]
            row['prefills'] = [engine.prefill(text, session) for text in versions]
        row.update(engine.translate(case['text'], Event(), session))
        row['tokens'] = tokens
        if case['id'] in references:
            ref = references[case['id']]
            row.update(identical_tokens=tokens == ref.get('tokens'),
                       identical_draft=row['draft'] == ref.get('draft'),
                       identical_translation=row['translation'] == ref.get('translation'))
    except Exception as exc:
        row.update(error=repr(exc), tokens=tokens)
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            rows.append(row)
            save()
            raise
    if case['id'] in references:
        ref = references[case['id']]
        row.update(identical_tokens=tokens == ref.get('tokens'),
                   identical_draft=row.get('draft') == ref.get('draft'),
                   identical_translation=row.get('translation') == ref.get('translation'),
                   identical_error=row.get('error') == ref.get('error'))
    row.update(wall_seconds=round(time.perf_counter()-start, 3), memory=memory())
    if getattr(engine, 'fast_execution', None):
        row['decoder_graphs'] = len(engine.fast_execution['decoder'].graphs)
    if len(rows) % 20 == 0:
        row['nvidia_smi'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,noheader'],
            text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip()
    rows.append(row)
    save()
    print(json.dumps({k: v for k, v in row.items() if k in
                      ['id', 'error', 'timings', 'identical_tokens', 'identical_draft',
                       'identical_translation', 'memory']}, ensure_ascii=False), flush=True)
    if args.fail_fast and ('error' in row or row.get('identical_tokens') is False
                          or row.get('identical_translation') is False):
        raise RuntimeError(f"Comparison failed for {case['id']}; see {report}")
document['completed'] = True
document['cuda_ooms'] = torch.cuda.memory_stats().get('num_ooms', 0)
save()
print('COMPLETE', len(rows), flush=True)
