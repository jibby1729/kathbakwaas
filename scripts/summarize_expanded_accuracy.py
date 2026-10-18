"""Summarize the expanded audit without overwriting the earlier evidence."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'artifacts'
corpus = json.loads((ART / 'expanded-corpus.json').read_text(encoding='utf-8'))
reports = {name: json.loads((ART / f'expanded-{name}.json').read_text(encoding='utf-8'))
           for name in ['stock', 'native', 'typing']}
baseline = {row['id']: row for row in reports['stock']['rows']}
labels = {'stock': 'Original NF4, four beams', 'native': 'Optimized, fresh prompt',
          'typing': 'Optimized, prepared prompt'}
assert all(r['completed'] and len(r['rows']) == len(corpus) for r in reports.values())
assert len({r['corpus_sha256'] for r in reports.values()}) == 1
assert all(r['source_sha256'] == reports['stock']['source_sha256'] for r in reports.values())
assert all(r['generation_options'] == reports['stock']['generation_options'] for r in reports.values())
lines = [
    '# Expanded inference equality audit', '',
    '400 new synthetic English inputs, with no overlap with the earlier 161 cases: '
    '160 individually authored sentences, 200 structured sentence combinations, and '
    '40 paragraphs assembled from eight openings and five endings. This is regression '
    'coverage, not a representative language benchmark or human quality evaluation.', '',
    f'Corpus: {sum(len(c["text"]) for c in corpus):,} characters; '
    f'{min(len(c["text"]) for c in corpus)}–{max(len(c["text"]) for c in corpus)} characters per input.', '',
    'All runs use the same pinned model revisions, stock NF4 weights, BF16 computation, '
    'four beams, length penalty 0.7, early stopping, 512-token output cap and CPU diacritizer. '
    'Each run starts a fresh process. The web server is stopped during the audit; '
    'neither completed-answer caching nor the browser’s curated Kashmir answer is used.', '',
    '| Path | Cases | Token differences | Draft differences | Diacritized differences | Errors | CUDA OOMs |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for name, report in reports.items():
    rows = report['rows']
    counts = [sum(r.get(k) is False for r in rows) for k in
              ['identical_tokens', 'identical_draft', 'identical_translation']]
    if name == 'stock':
        counts = ['reference'] * 3
    else:
        assert all('error' in r or all(k in r for k in
            ['identical_tokens', 'identical_draft', 'identical_translation']) for r in rows)
    lines.append(f'| {labels[name]} | {len(rows)} | ' + ' | '.join(map(str, counts))
                 + f' | {sum("error" in r for r in rows)} | {report.get("cuda_ooms", "unrecorded")} |')
lines += ['', 'The typing pass prepares a partial sentence, appends a word, deletes a word, '
          'then prepares the exact final input before translating. Full returned token sequences '
          'and both strings are compared exactly; no similarity threshold is used.', '',
          f'Prepared prompts actually reused: {sum(r.get("timings", {}).get("reused_prompt_tokens", 0) > 0 for r in reports["typing"]["rows"])}/{len(corpus)} typing cases.', '',
          '| Path | Median translation (s) | P95 translation (s) | Minimum sampled free GPU MiB |',
          '| --- | ---: | ---: | ---: |']
for name, report in reports.items():
    times = sorted(r['elapsed_seconds'] for r in report['rows'] if 'elapsed_seconds' in r)
    free = [r['memory']['free_mib'] for r in report['rows']]
    lines.append(f'| {labels[name]} | {statistics.median(times):.2f} | '
                 f'{times[math.ceil(len(times)*.95)-1]:.2f} | {min(free):.0f} |')
lines += ['', 'Timing includes generation and CPU diacritization. Typing preparation is separate, '
          'so its translation-stage latency is not end-to-end latency. Free GPU memory is sampled '
          'after each case, not continuously; CUDA OOM counters are cumulative for each fresh process.', '',
          '## Differences and errors', '']
differences = []
for name, report in reports.items():
    for row in report['rows']:
        if 'error' in row or row.get('identical_tokens') is False or row.get('identical_translation') is False:
            differences.append((name, row))
            lines += [f'### {name}: {row["id"]}', '', row['text'], '',
                      f'Stock: {baseline[row["id"]].get("translation", "error")}', '',
                      f'This run: {row.get("translation", row.get("error"))}', '']
if not differences:
    lines += ['None.', '']
lines += ['## Limits', '',
          'Passing this fixed set cannot prove equality for every possible input, software version, '
          'or device. Equality with the original model does not establish translation quality. '
          'The earlier rejected incremental-prefix cache is not used by the current application. '
          'The one curated browser answer intentionally overrides model inference and is separate '
          'from this audit. No Kashmir/Kashir spelling post-processing has been added.', '',
          '## Reproduction', '', '```powershell',
          '.venv/Scripts/python.exe scripts/build_expanded_corpus.py',
          '# Stop the local model server first; this runner restores it after the sequential runs.',
          '.venv/Scripts/python.exe scripts/run_expanded_accuracy.py',
          '.venv/Scripts/python.exe scripts/summarize_expanded_accuracy.py', '```', '',
          'Full evidence: expanded-corpus.json, expanded-stock.json, expanded-native.json, '
          'expanded-typing.json and expanded-status.json in artifacts/. Reports include source '
          'hashes, corpus hash, model revisions, library versions and effective decoding options.', '']
live_path = ART / 'expanded-live.json'
if live_path.exists():
    live = json.loads(live_path.read_text(encoding='utf-8'))
    lines += ['## Live web verification', '',
              'After restarting the model server, three fresh HTTP responses matched the stock '
              'draft and diacritized text. All had `cached: false`. The API does not expose token IDs.', '',
              '| Case | Preparation | Preparation HTTP (ms) | Translation HTTP (ms) | Prompt tokens reused | Matches stock |',
              '| --- | --- | ---: | ---: | ---: | --- |']
    for row in live:
        assert all(row['result'][key] == baseline[row['id']][key] for key in ['draft', 'translation'])
        lines.append(f'| {row["id"]} | {row["preparation"] or "none"} | '
                     f'{row.get("prefill_http_ms", 0)} | {row["translate_http_ms"]} | '
                     f'{row["result"]["timings"]["reused_prompt_tokens"]} | {row["matches_stock"]} |')
    lines += ['', 'Preparation is separate from the translation request. The partial preparation '
              'deliberately differs from the final paragraph, so that request must reuse zero prompt tokens. '
              'The page reported Ready and the updated name suggestion was verified in the browser.', '']
path = ART / 'expanded-accuracy-summary.md'
path.write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines[:25]))
