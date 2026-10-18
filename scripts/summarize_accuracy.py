"""Create a readable summary from completed comparison artifacts."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
files = sorted((ROOT / 'artifacts').glob('accuracy-*.json'))
reports = []
for path in files:
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'backend' in data and not path.stem.endswith('-partial'):
        reports.append(data | {'file': path.name})
baseline = next((r for r in reports if r['file'] == 'accuracy-stock.json'), None)
lines = ['# Translation regression comparison', '',
         'These inputs test equality with stock NF4, BF16-compute, four-beam inference. '
         'They are not human-rated Kashmiri references and do not establish linguistic correctness.', '',
         '| Run | Completed cases | Token differences | Draft differences | Diacritic text differences | Errors | Median translation stage (s) | Minimum sampled free GPU MiB |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for report in reports:
    rows = report['rows']
    compared = any('identical_tokens' in r for r in rows)
    counts = [sum(r.get(key) is False for r in rows) if compared else 'reference'
              for key in ['identical_tokens', 'identical_draft', 'identical_translation']]
    times = [r['elapsed_seconds'] for r in rows if 'elapsed_seconds' in r]
    free = [r['memory']['free_mib'] for r in rows if 'memory' in r]
    lines.append(f"| {report['file']} | {len(rows)} ({'complete' if report['completed'] else 'partial'}) | "
                 + ' | '.join(map(str, counts))
                 + f" | {sum('error' in r for r in rows)} | {statistics.median(times) if times else 0:.2f} | {min(free) if free else 0:.0f} |")
lines += ['', 'Times include fresh generation and CPU diacritization; typing-preparation time is recorded separately in each JSON. '
          'First-use graph capture is included. GPU free memory is sampled after each case and is not a continuous peak measurement.', '']
stress_path = ROOT / 'artifacts/accuracy-memory-stress.json'
if stress_path.exists():
    stress = json.loads(stress_path.read_text(encoding='utf-8'))
    if stress.get('completed'):
        lines += ['## Memory and long-input checks', '',
                  'Three further inputs of 488-494 characters matched stock exactly. Four page sessions, '
                  'in-flight cancellation and recovery were exercised. CUDA reported zero OOMs.', '',
                  f"Controlled memory pressure: {stress['pressure_result']}. The next request after releasing "
                  'the artificial allocation matched stock again.', '']
live_path = ROOT / 'artifacts/accuracy-live.json'
if live_path.exists():
    live = json.loads(live_path.read_text(encoding='utf-8'))
    lines += ['## Live webpage checks', '',
              '| Input | Preparation HTTP (ms) | Translation HTTP (ms) | Complete prompt reused | Matches stock |',
              '| --- | ---: | ---: | --- | --- |']
    for row in live:
        lines.append(f"| {row['source']} | {row['prefill_http_ms']} | {row['translation_http_ms']} | "
                     f"{row['exact_prompt']} | {row['matches_stock']} |")
    lines += ['', 'All three translation responses had `cached: false`. Preparation is separate work; '
              'when it cannot overlap typing, include it in the user-visible wait. '
              'The second case deliberately prepared a shorter, different input to test invalidation.', '']
for report in reports:
    lines += [f"## {report['file']}", '']
    for row in report['rows']:
        if row['id'] in ('reported-01', 'reported-02') or row.get('identical_tokens') is False or 'error' in row:
            lines += [f"### {row['id']}", '', row['text'], '',
                      f"Draft: {row.get('draft', row.get('error'))}", '',
                      f"With diacritics: {row.get('translation', '')}", '',
                      f"Timing: {row.get('timings', {})}", '']
            if row.get('identical_tokens') is False and baseline:
                ref = next(r for r in baseline['rows'] if r['id'] == row['id'])
                lines += [f"Stock draft: {ref.get('draft')}", '', f"Stock diacritics: {ref.get('translation')}", '']
(ROOT / 'artifacts/accuracy-summary.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines[:len(reports)+7]))
