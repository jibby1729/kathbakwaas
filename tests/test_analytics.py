import json
import sqlite3
from types import SimpleNamespace

import httpx
from starlette.testclient import TestClient

from app.analytics import TrafficStats, cloudflare_ip, summary
from app.sharing import create_app


def test_counts_persist_without_raw_ips_or_text(tmp_path):
    path = tmp_path / 'traffic.sqlite3'
    stats = TrafficStats(path)
    stats.record('page', 200, '192.0.2.1', 2)
    stats.record('page', 200, '192.0.2.1', 2)
    stats.record('translation', 200, '192.0.2.1', 10, False)
    stats.record('translation', 200, '2001:db8::1', 1, True)
    stats.record('translation', 500, '192.0.2.1', 1)
    stats.record('translation', 429, '192.0.2.1', 1)
    stats.record('translation', 499, '192.0.2.1', 1)
    stats.record('page', 200, '192.0.2.88', 1, check_token=stats.check_token)
    stats.close()
    data = summary(path)
    assert data['page_loads'] == 2
    assert data['approximate_unique_ips'] == 2
    assert data['successful_translations'] == 2
    assert data['cached_translations'] == data['fresh_translations'] == 1
    assert data['busy_translations'] == data['cancelled_translations'] == data['server_error_translations'] == 1
    assert summary(path, checks=True)['page_loads'] == 1
    with sqlite3.connect(path) as db:
        dump = '\n'.join(db.iterdump())
        assert '192.0.2.' not in dump and '2001:db8' not in dump
    reopened = TrafficStats(path)
    reopened.record('page', 200, '192.0.2.1', 2)
    reopened.close()
    assert summary(path)['approximate_unique_ips'] == 2
    assert summary(path)['page_loads'] == 3


def test_only_valid_cloudflare_header_from_loopback_is_used():
    def request(peer, ip, ray='edge-ray'):
        return SimpleNamespace(client=SimpleNamespace(host=peer), headers={
            'cf-connecting-ip': ip, 'cf-ray': ray, 'x-forwarded-for': '192.0.2.99'})
    assert cloudflare_ip(request('127.0.0.1', '::ffff:192.0.2.1')) == '192.0.2.1'
    assert cloudflare_ip(request('192.0.2.9', '192.0.2.1')) is None
    assert cloudflare_ip(request('127.0.0.1', '192.0.2.1', '')) is None
    assert cloudflare_ip(request('127.0.0.1', 'bad\n192.0.2.1')) is None


def test_gateway_ignores_prefills_assets_and_health_and_never_saves_text(tmp_path):
    async def upstream(request):
        return httpx.Response(200, json={'translation': 'private output', 'cached': False})
    path = tmp_path / 'traffic.sqlite3'
    app = create_app(transport=httpx.MockTransport(upstream), analytics_path=path)
    headers = {'CF-Connecting-IP': '192.0.2.1', 'CF-Ray': 'test-ray'}
    with TestClient(app, client=('127.0.0.1', 5555)) as client:
        assert client.get('/', headers=headers).status_code == 200
        for route in ['/api/health', '/static/app.js']:
            client.get(route, headers=headers)
        client.post('/api/prefill', json={'text': 'secret text'}, headers=headers)
        assert client.post('/api/translate', json={'text': 'secret text'}, headers=headers).status_code == 200
        client.get('/')  # local probes have no Cloudflare header
        client.get('/traffic.sqlite3', headers=headers)
        client.post('/api/translate', json={}, headers={**headers, 'Origin': 'https://unrelated.example'})
    data = summary(path)
    assert data['page_loads'] == 1 and data['successful_translations'] == 1
    assert data['approximate_unique_ips'] == 1
    with sqlite3.connect(path) as db:
        dump = '\n'.join(db.iterdump())
        assert 'secret text' not in dump and 'private output' not in dump
        assert db.execute("SELECT COUNT(*) FROM counters WHERE kind NOT IN ('page','translation')").fetchone()[0] == 0


def test_logging_failure_does_not_break_translation(tmp_path, monkeypatch):
    async def upstream(request):
        return httpx.Response(200, json={'translation': 'ok', 'cached': True})
    app = create_app(transport=httpx.MockTransport(upstream), analytics_path=tmp_path / 'traffic.sqlite3')
    with TestClient(app, client=('127.0.0.1', 1234)) as client:
        def broken(*args, **kwargs):
            raise sqlite3.OperationalError('database is locked')
        monkeypatch.setattr(app.app.state.analytics, 'record', broken)
        response = client.post('/api/translate', json={'text': 'hello'}, headers={
            'CF-Connecting-IP': '192.0.2.1', 'CF-Ray': 'test'})
        assert response.status_code == 200 and response.json()['translation'] == 'ok'


def test_detailed_events_bounded_and_old_events_expire(tmp_path):
    stats = TrafficStats(tmp_path / 'traffic.sqlite3')
    stats.db.executemany('INSERT INTO events(time,kind,status,cached,is_check,elapsed_ms) VALUES (?,?,?,?,?,?)',
        [('2000-01-01', 'page', 200, -1, 0, 1)] * 10002)
    stats.db.commit()
    stats.record('page', 200, '192.0.2.1', 1)
    assert stats.db.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 1
    stats.close()
