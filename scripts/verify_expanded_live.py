"""Verify fresh HTTP responses against the expanded stock references."""
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
refs = {r['id']: r for r in json.loads((ROOT / 'artifacts/expanded-stock.json').read_text(encoding='utf-8'))['rows']}

def post(endpoint, text):
    request = Request(f'http://127.0.0.1:8000/api/{endpoint}',
                      data=json.dumps({'text': text, 'session_id': 'expanded-live-verification'}).encode(),
                      headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urlopen(request, timeout=180) as response:
        data = json.load(response)
    return data, round((time.perf_counter() - start) * 1000, 2)

rows = []
for case_id, preparation in [('new_daily-01', None), ('new_emotions-04', 'exact'), ('new_paragraph-00-02', 'partial')]:
    ref = refs[case_id]
    row = {'id': case_id, 'source': ref['text'], 'preparation': preparation}
    if preparation:
        prefix = ref['text'] if preparation == 'exact' else ' '.join(ref['text'].split()[:5])
        row['prefill'], row['prefill_http_ms'] = post('prefill', prefix)
    row['result'], row['translate_http_ms'] = post('translate', ref['text'])
    row['matches_stock'] = all(row['result'][key] == ref[key] for key in ['draft', 'translation'])
    rows.append(row)
    (ROOT / 'artifacts/expanded-live.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    assert row['matches_stock'], case_id
    assert row['result']['cached'] is False, 'Expected fresh inference'
    reused = row['result']['timings']['reused_prompt_tokens']
    assert (reused > 0) if preparation == 'exact' else (reused == 0), 'Unexpected prefill reuse'
    print(json.dumps({'id': case_id, 'matches_stock': True, 'cached': False,
                      'reused_prompt_tokens': reused, 'translate_http_ms': row['translate_http_ms']}), flush=True)
