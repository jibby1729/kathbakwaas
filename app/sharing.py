"""Loopback-only gateway for the optional public tunnel; never imports models."""
import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import os
from pathlib import Path
import time

import httpx
from starlette.applications import Starlette
from starlette.requests import Request, ClientDisconnect
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from .analytics import TrafficStats, cloudflare_ip

MAX_BODY = 16 * 1024
LIMITS = {"/api/translate": 3, "/api/prefill": 1, "/api/speech": 1}


class Bucket:
    def __init__(self, rate=4, burst=24):
        self.rate, self.burst = rate, burst
        self.tokens, self.updated = float(burst), time.monotonic()

    def take(self):
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True


def create_app(upstream="http://127.0.0.1:8000", transport=None, analytics_path=None):
    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            base_url=upstream, transport=transport, trust_env=False,
            timeout=httpx.Timeout(190, connect=3),
            limits=httpx.Limits(max_connections=32),
        ) as client:
            app.state.client = client
            app.state.analytics = TrafficStats(analytics_path) if analytics_path else None
            try:
                yield
            finally:
                if app.state.analytics:
                    app.state.analytics.close()

    async def proxy(request: Request):
        path = request.url.path
        if "\\" in path or any(part in (".", "..") for part in path.split("/")):
            return JSONResponse({"detail": "Not found."}, status_code=404)
        if request.method == "GET" and path == "/robots.txt":
            return Response("User-agent: *\nDisallow: /\n", media_type="text/plain")
        allowed = (
            request.method == "GET" and (path in ("/", "/api/health") or path.startswith("/static/"))
        ) or (request.method == "POST" and path in LIMITS)
        if not allowed:
            return JSONResponse({"detail": "Not found."}, status_code=404)
        # Global limits cannot be bypassed by forged IP headers or new sessions.
        if app.state.active >= 32:
            return busy()
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and origin not in (f"https://{request.headers.get('host')}",
                                         f"http://{request.headers.get('host')}"):
                return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
            if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                return JSONResponse({"detail": "Send JSON."}, status_code=415)
            if not app.state.bucket.take() or app.state.pending[path] >= LIMITS[path]:
                return busy()
        app.state.active += 1
        if path in LIMITS:
            app.state.pending[path] += 1
        try:
            body = bytearray()
            try:
                async with asyncio.timeout(5):
                    async for chunk in request.stream():
                        body.extend(chunk)
                        if len(body) > MAX_BODY:
                            return JSONResponse({"detail": "Request too large."}, status_code=413)
            except TimeoutError:
                return JSONResponse({"detail": "Request timed out."}, status_code=408)
            except ClientDisconnect:
                return Response(status_code=499)

            async def disconnected():
                # Body is fully consumed. Await disconnect directly; Starlette's
                # polling helper can swallow a task cancellation in its cancel scope.
                while True:
                    message = await request.receive()
                    if message["type"] == "http.disconnect":
                        return

            # Forward only intended fields, never cookies, arbitrary headers or destinations.
            # Cancelling the upstream socket lets the model stop obsolete translations.
            work = asyncio.create_task(app.state.client.request(
                request.method, path, content=bytes(body),
                headers={"Content-Type": "application/json"} if request.method == "POST" else {},
            ))
            watcher = asyncio.create_task(disconnected())
            try:
                done, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
                if watcher in done:
                    return Response(status_code=499)
                result = await work
                headers = {key: value for key, value in result.headers.items() if key.lower() in {
                    "content-type", "content-security-policy",
                    "retry-after", "x-audio-cached",
                }}
                return Response(result.content, status_code=result.status_code, headers=headers)
            except httpx.HTTPError:
                return JSONResponse({"detail": "The translator is unavailable. Please try again."}, status_code=503)
            finally:
                for task in (watcher, work):
                    task.cancel()
                for task in (watcher, work):
                    with suppress(asyncio.CancelledError, Exception):
                        await task
        finally:
            app.state.active -= 1
            if path in LIMITS:
                app.state.pending[path] -= 1

    async def observed_proxy(request):
        kind = ('page' if request.method == 'GET' and request.url.path == '/' else
                'translation' if request.method == 'POST' and request.url.path == '/api/translate' else None)
        started, status, cached = time.perf_counter(), 499, None
        try:
            response = await proxy(request)
            status = response.status_code
            if kind == 'translation' and status == 200:
                import json
                with suppress(ValueError, AttributeError):
                    cached = json.loads(response.body).get('cached')
            return response
        except Exception:
            status = 500
            raise
        finally:
            if kind and app.state.analytics:
                try:
                    app.state.analytics.record(kind, status, cloudflare_ip(request),
                        round((time.perf_counter() - started) * 1000, 2), cached,
                        request.headers.get('x-kath-verification', ''))
                except Exception:
                    # Analytics must not turn a successful translation into an error.
                    logging.getLogger(__name__).warning('Traffic count could not be saved', exc_info=True)

    app = Starlette(routes=[Route("/{path:path}", observed_proxy, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])], lifespan=lifespan)
    app.state.analytics = None
    app.state.active = 0
    app.state.pending = dict.fromkeys(LIMITS, 0)
    app.state.bucket = Bucket()
    return PublicHeaders(app)


def busy():
    return JSONResponse({"detail": "The translator is busy. Please try again in a moment."},
                        status_code=429, headers={"Retry-After": "2"})


class PublicHeaders:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def wrapped(message):
            if message["type"] == "http.response.start":
                message["headers"] += [
                    (b"x-robots-tag", b"noindex, nofollow, noarchive"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                ]
            await send(message)
        await self.app(scope, receive, wrapped)


app = create_app(analytics_path=Path(os.getenv('KOSHUR_ANALYTICS_DB',
    str(Path(__file__).resolve().parent.parent / '.hosting' / 'traffic.sqlite3'))))
