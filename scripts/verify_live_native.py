"""Local HTTP check of fresh translations and typing-time prefill."""
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import URLError
import uuid

ROOT = Path(__file__).resolve().parent.parent
BASE = 'http://127.0.0.1:8000'


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=60) as response:
        return json.load(response)


deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    try:
        health = request('/api/health')
        if health['state'] == 'ready':
            break
        if health['state'] == 'error':
            raise RuntimeError(health)
    except URLError:
        pass
    time.sleep(1)
else:
    raise RuntimeError('Server did not become ready')

refs = {row['source']: row for row in json.loads((ROOT / 'artifacts/full-pipeline-checks.json').read_text(encoding='utf-8')) if row['phase'] == 'baseline'}
source = next(iter(refs))
session = str(uuid.uuid4())
prefills = [request('/api/prefill', {'text': text, 'session_id': session})
            for text in ['oh my', 'oh my god you', source]]
rows = []
for text in [source, 'How are you?', 'Please close the window.']:
    started = time.perf_counter()
    result = request('/api/translate', {'text': text, 'session_id': session})
    row = {'source': text, 'http_ms': round((time.perf_counter() - started) * 1000, 2),
           'identical_text': result['draft'] == refs[text]['draft'] and result['translation'] == refs[text]['translation'], **result}
    rows.append(row)
    print(json.dumps({key: value for key, value in row.items() if key not in ('source', 'draft', 'translation')}), flush=True)
    assert row['identical_text']
    assert result['cached'] is False, 'This HTTP response was cached; do not count it as a fresh timing'
(ROOT / 'artifacts/native-live-verification.json').write_text(
    json.dumps({'prefills': prefills, 'translations': rows}, ensure_ascii=False, indent=2), encoding='utf-8',
)
print('LIVE_VERIFIED', flush=True)
