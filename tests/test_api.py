"""Contract and scheduler tests use a stand-in engine; real GPU checks are separate."""
import asyncio
import threading
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from app.main import app
from app.engine import TranslationCancelled


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("KOSHUR_SKIP_MODEL_LOAD", "1")
    with TestClient(app) as client:
        yield client


class FakeEngine:
    state = "ready"
    calls = 0

    def translate(self, text, cancelled, session=None):
        self.calls += 1
        return {"translation": "کٲشُر", "draft": "کاشر", "elapsed_seconds": 0.01}


def test_real_page_and_local_assets(client):
    page = client.get("/")
    assert page.status_code == 200
    assert 'lang="ks-Arab" dir="rtl"' in page.text
    for asset in ["styles.css", "app.js", "flower.svg", "fonts/noto-naskh-arabic.ttf"]:
        assert client.get("/static/" + asset).status_code == 200
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]


def test_preview_never_fakes_a_translation(client):
    assert client.get("/api/health").json()["state"] == "offline"
    assert client.post("/api/translate", json={"text": "Hello"}).status_code == 503


@pytest.mark.parametrize("text", ["", "  \n ", "x" * 501])
def test_invalid_text_is_rejected(client, text):
    assert client.post("/api/translate", json={"text": text}).status_code == 422


def test_pipeline_response_and_bounded_cache(client, monkeypatch):
    monkeypatch.setattr("app.main.CACHE_SIZE", 3)
    engine = FakeEngine()
    app.state.engine = engine
    first = client.post("/api/translate", json={"text": " Hello "}).json()
    again = client.post("/api/translate", json={"text": "Hello"}).json()
    assert first["translation"] == "کٲشُر"
    assert first["draft"] == "کاشر"
    assert first["cached"] is False
    assert again["cached"] is True
    assert engine.calls == 1
    for index in range(2):
        client.post("/api/translate", json={"text": f"Sentence {index}"})
    assert client.post("/api/translate", json={"text": "Hello"}).json()["cached"] is True
    client.post("/api/translate", json={"text": "Sentence 2"})
    assert len(app.state.cache) == 3
    assert "Hello" in app.state.cache
    assert "Sentence 0" not in app.state.cache


def test_queue_is_bounded(client):
    app.state.engine = FakeEngine()
    app.state.pending = 8
    response = client.post("/api/translate", json={"text": "Hello"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "2"


def test_engine_errors_are_recoverable(client):
    class BrokenEngine(FakeEngine):
        def translate(self, text, cancelled):
            raise ValueError("Please try a shorter sentence.")

    app.state.engine = BrokenEngine()
    response = client.post("/api/translate", json={"text": "Hello"})
    assert response.status_code == 422
    assert response.json()["detail"] == "Please try a shorter sentence."
    assert app.state.pending == 0


def test_gpu_memory_pressure_defers_request_and_releases_queue(client):
    from app.gpu_memory import GPUMemoryBusy

    class BusyGPU(FakeEngine):
        def translate(self, text, cancelled):
            raise GPUMemoryBusy('GPU memory is busy.')

    app.state.engine = BusyGPU()
    response = client.post('/api/translate', json={'text': 'Hello'})
    assert response.status_code == 503
    assert response.headers['retry-after'] == '2'
    assert app.state.pending == 0
    assert not app.state.cache


def test_prefill_requires_page_session_and_yields_to_translation(client):
    class PrefillEngine(FakeEngine):
        def prefill(self, text, session):
            return {"prefilled": True, "session": session}

    app.state.engine = PrefillEngine()
    assert client.post("/api/prefill", json={"text": "Hello"}).status_code == 400
    payload = {"text": "Hello", "session_id": "page-1"}
    assert client.post("/api/prefill", json=payload).json()["session"] == "page-1"
    app.state.pending = 1
    assert client.post("/api/prefill", json=payload).json() == {"prefilled": False, "reason": "busy"}


def test_failed_prefill_does_not_prevent_translation(client):
    class FailedPrefillEngine(FakeEngine):
        def prefill(self, text, session):
            raise RuntimeError("Cache unavailable")

    app.state.engine = FailedPrefillEngine()
    payload = {"text": "Hello", "session_id": "page-1"}
    assert client.post("/api/prefill", json=payload).json() == {"prefilled": False}
    assert client.post("/api/translate", json=payload).status_code == 200


def test_cancelled_prefill_keeps_lock_until_forward_finishes(monkeypatch):
    class PrefillEngine(FakeEngine):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def prefill(self, text, session):
            self.started.set()
            assert self.release.wait(2)
            return {"prefilled": True}

        def translate(self, text, cancelled, session=None):
            assert self.release.is_set(), "Translation overlapped speculative prefill"
            return super().translate(text, cancelled, session)

    async def run():
        monkeypatch.setenv("KOSHUR_SKIP_MODEL_LOAD", "1")
        async with app.router.lifespan_context(app):
            engine = PrefillEngine()
            app.state.engine = engine
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                payload = {"text": "Hello", "session_id": "page-1"}
                first = asyncio.create_task(client.post("/api/prefill", json=payload))
                assert await asyncio.to_thread(engine.started.wait, 2)
                first.cancel()
                second = asyncio.create_task(client.post("/api/translate", json=payload))
                await asyncio.sleep(0.03)
                assert not second.done()
                engine.release.set()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert (await second).status_code == 200

    asyncio.run(run())


def test_cancellation_releases_gpu_before_next_request(monkeypatch):
    """Task cancellation must keep the inference lock until the thread has stopped."""
    class SlowEngine(FakeEngine):
        def __init__(self):
            self.active = 0
            self.peak = 0
            self.started = threading.Event()

        def translate(self, text, cancelled):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                if text == "First":
                    self.started.set()
                    while not cancelled.is_set():
                        time.sleep(0.005)
                    time.sleep(0.05)
                    raise TranslationCancelled()
                return super().translate(text, cancelled)
            finally:
                self.active -= 1

    async def run():
        monkeypatch.setenv("KOSHUR_SKIP_MODEL_LOAD", "1")
        async with app.router.lifespan_context(app):
            engine = SlowEngine()
            app.state.engine = engine
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/api/translate", json={"text": "First"}))
                assert await asyncio.to_thread(engine.started.wait, 2)
                first.cancel()
                second = asyncio.create_task(client.post("/api/translate", json={"text": "Second"}))
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert (await second).status_code == 200
                assert engine.peak == 1
                assert app.state.pending == 0

    asyncio.run(run())
