import asyncio
import pytest

from app.main import app
from app.speech import SpeechBusy, SpeechService
from app.tts_worker import split_text
from fastapi.testclient import TestClient


def test_chunking_keeps_text_and_bounds_each_call():
    text = ('کٲشُر زَبان ' * 80).strip()
    chunks = split_text(text)
    assert ' '.join(chunks) == text
    assert all(len(c) <= 180 for c in chunks)
    assert split_text('چون ناو کیا چھُ؟ بہٕ چھُس ٹھیک۔') == ['چون ناو کیا چھُ؟', 'بہٕ چھُس ٹھیک۔']
    with pytest.raises(ValueError):
        split_text('ک' * 181)


def test_speech_validation_and_response(monkeypatch):
    monkeypatch.setenv('KOSHUR_SKIP_MODEL_LOAD', '1')
    with TestClient(app) as client:
        for data in [dict(text='', voice='male'), dict(text='hello', voice='male'),
                     dict(text='کٲشُر', voice='unknown'), dict(text='ک' * 1501, voice='female')]:
            assert client.post('/api/speech', json=data).status_code == 422
        calls = []

        async def fake_speak(text, voice):
            calls.append((text, voice))
            return b'RIFFtest', False

        monkeypatch.setattr(app.state.speech, 'speak', fake_speak)
        response = client.post('/api/speech', json={'text': ' کٲشُر ', 'voice': 'female'})
        assert response.status_code == 200
        assert response.content == b'RIFFtest'
        assert response.headers['content-type'] == 'audio/wav'
        assert "media-src 'self' blob:" in response.headers['content-security-policy']
        assert calls == [('کٲشُر', 'female')]


def test_cache_separates_voices_and_bounds_memory():
    async def check():
        service = SpeechService()
        calls = []

        async def generate(text, voice):
            calls.append((text, voice))
            return b'RIFF' + bytes(600_000), {'seconds': .1, 'duration': 1.0}

        service._generate = generate
        await service.speak('کٲشُر', 'male')
        assert (await service.speak('کٲشُر', 'male'))[1] is True
        assert (await service.speak('کٲشُر', 'female'))[1] is False
        assert len(calls) == 2
        for i in range(40):
            await service.speak(str(i), 'male')
        assert len(service.cache) <= 32
        assert service.cache_bytes <= 16 * 1024 * 1024
        async with service.lock:
            with pytest.raises(SpeechBusy):
                await service.speak('new text', 'female')

    asyncio.run(check())


def test_cancelled_request_keeps_worker_lock_until_response_consumed():
    async def check():
        service = SpeechService()
        started, release = asyncio.Event(), asyncio.Event()

        async def generate(text, voice):
            started.set()
            await release.wait()
            return b'RIFF', {'seconds': .1, 'duration': 1}

        service._generate = generate
        task = asyncio.create_task(service.speak('کٲشُر', 'male'))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert service.lock.locked()
        with pytest.raises(SpeechBusy):
            await service.speak('کٲشُر', 'female')
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not service.lock.locked()

    asyncio.run(check())
