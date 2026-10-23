import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from app.translation_cache import TranslationCacheStore


RESULT = {'translation': 'کٲشُر', 'draft': 'کاشر', 'elapsed_seconds': 1.2}


def test_disk_cache_preserves_results_and_lru_across_restart(tmp_path):
    path = tmp_path / 'translations.sqlite3'
    store = TranslationCacheStore(path, 'version1')
    for text in ('one', 'two', 'three'):
        store.put(text, RESULT, 3)
    store.touch('one')
    store.put('four', RESULT, 3)
    store.close()
    store = TranslationCacheStore(path, 'version1')
    assert list(store.load(3)) == ['three', 'one', 'four']
    assert store.load(3)['one'] == RESULT
    assert list(store.load(2)) == ['one', 'four']
    store.close()


def test_model_change_invalidates_disk_cache(tmp_path):
    path = tmp_path / 'cache.sqlite3'
    store = TranslationCacheStore(path, 'old-model')
    store.put('Hello', RESULT, 4096)
    store.close()
    store = TranslationCacheStore(path, 'new-model')
    assert not store.load(4096)
    store.close()


def test_unwritable_cache_falls_back_to_ram(tmp_path):
    store = TranslationCacheStore(tmp_path, 'version1')
    assert not store.load(4096)
    store.put('Hello', RESULT, 4096)
    store.touch('Hello')
    store.close()


def test_bad_record_is_skipped_and_cli_exports_unicode(tmp_path):
    path = tmp_path / 'cache.sqlite3'
    store = TranslationCacheStore(path, 'version1')
    store.put('Hello', RESULT, 4096)
    with store.connection:
        store.connection.execute('INSERT INTO translations VALUES (?, ?, ?)', ('bad', 'not-json', 2))
    assert list(store.load(4096)) == ['Hello']
    output = tmp_path / 'export.json'
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'cache.py'
    subprocess.run([sys.executable, str(script), '--database', str(path), '--export', str(output)], check=True)
    data = json.loads(output.read_text(encoding='utf-8'))
    assert data['count'] == 1
    assert data['translations'] == [{'english': 'Hello', 'translation': RESULT['translation'], 'draft': RESULT['draft']}]
    store.close()


def test_api_restores_cache_without_running_inference(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setenv('KOSHUR_SKIP_MODEL_LOAD', '1')
    monkeypatch.setenv('KOSHUR_CACHE_DB', str(tmp_path / 'api.sqlite3'))

    class Engine:
        state = 'ready'
        calls = 0

        def translate(self, text, cancelled, session=None):
            self.calls += 1
            return RESULT

    engine = Engine()
    with TestClient(app) as client:
        app.state.engine = engine
        assert client.post('/api/translate', json={'text': 'Hello'}).json()['cached'] is False
    with TestClient(app) as client:
        app.state.engine = engine
        result = client.post('/api/translate', json={'text': 'Hello'}).json()
        assert result == {**RESULT, 'cached': True}
        assert engine.calls == 1
        assert client.get('/api/cache').status_code == 404
        assert client.get('/static/../.hosting/translations.sqlite3').status_code == 404
