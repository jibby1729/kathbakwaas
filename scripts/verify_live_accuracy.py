"""Fresh HTTP checks against stock references, including exact-input invalidation."""
import json
from pathlib import Path
import time

import httpx

ROOT = Path(__file__).resolve().parent.parent
references = {row['id']: row for row in json.loads(
    (ROOT / 'artifacts/accuracy-stock.json').read_text(encoding='utf-8'))['rows']}
rows = []
with httpx.Client(base_url='http://127.0.0.1:8000', timeout=180) as client:
    for case_id, exact in [('reported-01', True), ('reported-02', False), ('everyday-02', True)]:
        ref = references[case_id]
        session = 'live-accuracy-' + case_id
        text = ref['text'] if exact else 'oh my god'
        started = time.perf_counter()
        response = client.post('/api/prefill', json={'text': text, 'session_id': session})
        response.raise_for_status()
        prefill = response.json()
        prefill_wall = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        response = client.post('/api/translate', json={'text': ref['text'], 'session_id': session})
        response.raise_for_status()
        result = response.json()
        wall = (time.perf_counter() - started) * 1000
        assert result['cached'] is False
        assert result['draft'] == ref['draft'] and result['translation'] == ref['translation']
        assert bool(result['timings']['reused_prompt_tokens']) == exact
        row = {'id': case_id, 'source': ref['text'], 'exact_prompt': exact,
               'prefill': prefill, 'prefill_http_ms': round(prefill_wall, 2),
               'translation_http_ms': round(wall, 2), 'matches_stock': True, **result}
        rows.append(row)
        print(case_id, round(prefill_wall, 2), round(wall, 2), result['timings'], flush=True)
(ROOT / 'artifacts/accuracy-live.json').write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
