import asyncio

import httpx
from starlette.testclient import TestClient

from app.sharing import Bucket, MAX_BODY, create_app


def fixture_app(handler=None):
    seen = []

    async def upstream(request):
        seen.append(request)
        if handler:
            return await handler(request)
        return httpx.Response(200, json={"translation": "unchanged"}, headers={
            "content-security-policy": "default-src 'self'", "x-audio-cached": "true",
        })

    return create_app(transport=httpx.MockTransport(upstream)), seen


def test_proxy_preserves_payload_and_output_without_forwarding_credentials():
    app, seen = fixture_app()
    with TestClient(app) as client:
        response = client.post('/api/translate', json={"text": "hello", "session_id": "test"},
                               headers={"Cookie": "secret=private", "Authorization": "private"})
        assert response.json() == {"translation": "unchanged"}
        assert response.headers['content-security-policy'] == "default-src 'self'"
        assert response.headers['x-robots-tag'].startswith('noindex')
        assert response.headers['referrer-policy'] == 'no-referrer'
        assert response.headers['cache-control'] == 'no-store'
        assert seen[0].url.host == '127.0.0.1'
        assert b'hello' in seen[0].content
        assert 'cookie' not in seen[0].headers
        assert 'authorization' not in seen[0].headers


def test_block_unintended_routes_cross_site_and_non_json():
    app, seen = fixture_app()
    with TestClient(app) as client:
        for path in ['/openapi.json', '/docs', '/.env', '/static/%2e%2e/openapi.json']:
            assert client.get(path).status_code == 404
        assert client.post('/api/translate', json={"text": "x"}, headers={"Origin": "https://evil.test"}).status_code == 403
        assert client.post('/api/translate', content='x').status_code == 415
        assert client.delete('/api/translate').status_code == 404
        assert not seen
        assert 'Disallow: /' in client.get('/robots.txt').text


def test_body_limit_including_chunked_upload():
    app, seen = fixture_app()
    with TestClient(app) as client:
        response = client.post('/api/translate', content=iter([b'x' * MAX_BODY, b'x']),
                               headers={'Content-Type': 'application/json'})
        assert response.status_code == 413
        assert not seen
        assert app.app.state.active == 0
        assert app.app.state.pending['/api/translate'] == 0


def test_rate_limit_cannot_be_bypassed_with_fake_ip():
    app, seen = fixture_app()
    app.app.state.bucket = Bucket(rate=0, burst=1)
    with TestClient(app) as client:
        assert client.post('/api/prefill', json={"text": "x"}).status_code == 200
        response = client.post('/api/translate', json={"text": "x"}, headers={'CF-Connecting-IP': '1.2.3.4'})
        assert response.status_code == 429
        assert response.headers['retry-after'] == '2'
        assert client.get('/').status_code == 200


def test_upstream_failure_is_generic_and_releases_slots():
    async def fail(request):
        raise httpx.ConnectError('private local path')
    app, _ = fixture_app(fail)
    with TestClient(app) as client:
        response = client.post('/api/speech', json={"text": "x"})
        assert response.status_code == 503
        assert 'private local path' not in response.text
        assert app.app.state.pending['/api/speech'] == 0


def test_concurrency_limit_and_disconnect_cancel_upstream():
    async def check():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def slow(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        app, _ = fixture_app(slow)
        disconnect = asyncio.Event()
        first_body = True

        async def receive():
            nonlocal first_body
            if first_body:
                first_body = False
                return {'type': 'http.request', 'body': b'{}', 'more_body': False}
            await disconnect.wait()
            return {'type': 'http.disconnect'}

        async def send(message):
            pass

        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                 'method': 'POST', 'scheme': 'http', 'path': '/api/prefill', 'query_string': b'',
                 'headers': [(b'host', b'test'), (b'content-type', b'application/json')],
                 'server': ('test', 80), 'client': ('127.0.0.1', 1234)}
        async with app.app.router.lifespan_context(app.app):
            task = asyncio.create_task(app(scope, receive, send))
            await asyncio.wait_for(entered.wait(), 2)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.post('/api/prefill', json={})
                assert response.status_code == 429
            disconnect.set()
            await asyncio.wait_for(task, 2)
            assert cancelled.is_set()
            assert app.app.state.active == 0
            assert app.app.state.pending['/api/prefill'] == 0
    asyncio.run(check())
