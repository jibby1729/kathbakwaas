"""Real CPU speech smoke test, with timing and saved WAVs for listening."""
import asyncio
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.speech import SpeechService


async def main():
    service = SpeechService()
    report = {'cases': []}
    try:
        started = time.perf_counter()
        await service.start()
        report.update(service.metadata, load_seconds=time.perf_counter() - started)
        for label, text in [('short', 'چون ناو کیا چھُ؟'),
                            ('phrase', 'کٔشیٖر چھِ کٲشِرؠن ہٕنٛز۔'),
                            ('long', 'بہٕ چھُس اَز واریاہ بۆچھ تہٕ مے چھُ تۆہہِ ہٕنٛدِس کتھہِ پؠٹھ ناراضگی۔')]:
            for voice in ['male', 'female']:
                start = time.perf_counter()
                audio, cached = await service.speak(text, voice)
                seconds = time.perf_counter() - start
                with wave.open(io.BytesIO(audio)) as wav:
                    assert wav.getframerate() == 22050 and wav.getnchannels() == 1
                    duration = wav.getnframes() / wav.getframerate()
                target = ROOT / f'artifacts/bolbosh-{label}-{voice}.wav'
                target.write_bytes(audio)
                item = dict(case=label, voice=voice, seconds=seconds, duration=duration,
                            cached=cached, sha256=hashlib.sha256(audio).hexdigest())
                report['cases'].append(item)
                print(json.dumps(item), flush=True)
        assert (await service.speak('چون ناو کیا چھُ؟', 'male'))[1]
        (ROOT / 'artifacts/bolbosh-check.json').write_text(json.dumps(report, indent=2))
    finally:
        await service.close()


if __name__ == '__main__':
    asyncio.run(main())
