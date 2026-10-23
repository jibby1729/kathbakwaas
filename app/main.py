import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
import logging
import os
from pathlib import Path
from threading import Event
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.datastructures import MutableHeaders

from .engine import TranslationCancelled, TranslationEngine
from .gpu_memory import GPUMemoryBusy
from .speech import SpeechBusy, SpeechService
from .translation_cache import TranslationCacheStore, cache_signature

ROOT = Path(__file__).resolve().parent.parent
MAX_CHARS = 500
# Completed translations live in CPU RAM and are evicted least recently used first.
CACHE_SIZE = 4096
logger = logging.getLogger(__name__)


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[a-zA-Z0-9-]+$")

    @field_validator("text")
    @classmethod
    def clean_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Please enter some English text.")
        return value


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1500)
    voice: Literal['male', 'female']

    @field_validator('text')
    @classmethod
    def clean_text(cls, value):
        value = value.strip()
        if not value or not any('\u0620' <= char <= '\u06ff' and char.isalpha() for char in value):
            raise ValueError('Please translate some text first.')
        return value


@asynccontextmanager
async def lifespan(app):
    app.state.engine = TranslationEngine()
    app.state.inference_lock = asyncio.Lock()
    app.state.cache = OrderedDict()
    app.state.cache_store = None
    # Preview/tests stay isolated unless an explicit test database is provided.
    cache_path = os.getenv('KOSHUR_CACHE_DB')
    if cache_path != '' and (cache_path or os.getenv('KOSHUR_SKIP_MODEL_LOAD') != '1'):
        app.state.cache_store = TranslationCacheStore(
            cache_path or ROOT / '.hosting' / 'translations.sqlite3', cache_signature(ROOT))
        app.state.cache = app.state.cache_store.load(CACHE_SIZE)
    app.state.pending = 0
    app.state.speech = SpeechService()
    loader = None
    speech_loader = None
    if os.getenv("KOSHUR_SKIP_MODEL_LOAD") == "1":
        app.state.engine.state = "offline"
        app.state.engine.message = "Preview only. Start the model service to translate."
    else:
        loader = asyncio.create_task(asyncio.to_thread(app.state.engine.load))
        async def warm_speech():
            await asyncio.shield(loader)
            try:
                async with app.state.speech.lock:
                    await app.state.speech.start()
            except Exception:
                logger.exception('Speech could not preload; translation remains available')
        speech_loader = asyncio.create_task(warm_speech())
    yield
    if speech_loader:
        speech_loader.cancel()
        with suppress(asyncio.CancelledError):
            await speech_loader
    if loader:
        await loader
    await app.state.speech.close()
    if app.state.cache_store:
        app.state.cache_store.close()


class SecurityHeadersMiddleware:
    """Pure ASGI middleware preserves cancellation and disconnect notifications."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                headers["Content-Security-Policy"] = (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
                    "media-src 'self' blob:; "
                    "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
                )
                if scope.get("path", "").startswith("/api/"):
                    headers["Cache-Control"] = "no-store"
            await send(message)
        await self.app(scope, receive, send_with_headers)


app = FastAPI(title="Kath Bath", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(SecurityHeadersMiddleware)


@app.get("/api/health")
async def health():
    return app.state.engine.status()


@app.post('/api/speech')
async def speech(body: SpeechRequest):
    try:
        audio, cached = await app.state.speech.speak(body.text, body.voice)
        return Response(audio, media_type='audio/wav', headers={'X-Audio-Cached': str(cached).lower()})
    except SpeechBusy as exc:
        raise HTTPException(429, str(exc), headers={'Retry-After': '2'}) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception('Speech failed')
        raise HTTPException(503, 'Audio could not be made. Please try again.') from exc


@app.post("/api/translate")
async def translate(body: TranslateRequest, request: Request):
    engine = app.state.engine
    if engine.state != "ready":
        raise HTTPException(503, engine.message, headers={"Retry-After": "5"})
    cache = app.state.cache
    if body.text in cache:
        cache.move_to_end(body.text)
        if app.state.cache_store:
            app.state.cache_store.touch(body.text)
        return {**cache[body.text], "cached": True}
    # One GPU worker; bound pending requests to avoid a growing inference queue.
    if app.state.pending >= 8:
        raise HTTPException(429, "The translator is busy. Please try again in a moment.", headers={"Retry-After": "2"})
    app.state.pending += 1
    cancelled = Event()

    async def watch_disconnect():
        while not cancelled.is_set():
            if await request.is_disconnected():
                cancelled.set()
                return
            await asyncio.sleep(0.1)

    watcher = asyncio.create_task(watch_disconnect())
    worker = None
    try:
        async with app.state.inference_lock:
            if cancelled.is_set():
                return JSONResponse({"detail": "Translation cancelled."}, status_code=499)
            # A preceding request may have populated the cache while we waited.
            if body.text in cache:
                cache.move_to_end(body.text)
                if app.state.cache_store:
                    app.state.cache_store.touch(body.text)
                return {**cache[body.text], "cached": True}
            options = {"session": body.session_id} if body.session_id else {}
            worker = asyncio.create_task(asyncio.to_thread(engine.translate, body.text, cancelled, **options))
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled.set()
                # Keep the GPU lock until the cancelled generation actually exits.
                with suppress(TranslationCancelled):
                    await worker
                raise
            if cancelled.is_set():
                return JSONResponse({"detail": "Translation cancelled."}, status_code=499)
            cache[body.text] = result
            while len(cache) > CACHE_SIZE:
                cache.popitem(last=False)
            if app.state.cache_store:
                app.state.cache_store.put(body.text, result, CACHE_SIZE)
            return {**result, "cached": False}
    except TranslationCancelled:
        return JSONResponse({"detail": "Translation cancelled."}, status_code=499)
    except GPUMemoryBusy as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "2"}) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Translation failed")
        raise HTTPException(500, "The translator had a server error. Please try again in a moment.") from exc
    finally:
        app.state.pending -= 1
        cancelled.set()
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


@app.post("/api/prefill")
async def prefill(body: TranslateRequest):
    if app.state.engine.state != "ready":
        raise HTTPException(503, app.state.engine.message)
    if not body.session_id:
        raise HTTPException(400, "A page session is required for prefill.")
    # Speculative input processing must never queue ahead of a real translation.
    if app.state.pending or app.state.inference_lock.locked():
        return {"prefilled": False, "reason": "busy"}
    async with app.state.inference_lock:
        worker = asyncio.create_task(asyncio.to_thread(app.state.engine.prefill, body.text, body.session_id))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # A single model forward cannot be interrupted halfway through a layer.
            # Hold the lock until it exits, even if the browser cancels its fetch.
            with suppress(Exception):
                await worker
            raise
        except Exception:
            logger.exception("Background prefill failed; normal translation is still available")
            return {"prefilled": False}


@app.get("/")
async def index():
    return FileResponse(ROOT / "web" / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
