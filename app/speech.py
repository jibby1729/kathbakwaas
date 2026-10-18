"""Bounded speech service using a separate CPU-only Python environment."""
import asyncio
import base64
from collections import OrderedDict
import json
import logging
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


class SpeechBusy(Exception):
    pass


class SpeechService:
    def __init__(self):
        self.process = None
        self.lock = asyncio.Lock()
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.log_file = None
        self.metadata = None

    async def start(self):
        if self.process and self.process.returncode is None:
            return
        executable = ROOT / '.venv-tts' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not executable.exists():
            raise RuntimeError('Speech is not installed. Run the Bolbosh setup in README.md.')
        (ROOT / 'artifacts').mkdir(exist_ok=True)
        if self.log_file:
            self.log_file.close()
        self.log_file = (ROOT / 'artifacts/tts-worker.log').open('ab')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'PYTHONIOENCODING': 'utf-8',
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        self.process = await asyncio.create_subprocess_exec(
            str(executable), '-u', str(ROOT / 'app/tts_worker.py'),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=self.log_file, env=env, cwd=ROOT, limit=16 * 1024 * 1024,
            **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}),
        )
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), 90)
            self.metadata = json.loads(line)
            if not self.metadata.get('ready'):
                raise RuntimeError('Speech failed to start.')
        except BaseException:
            await self.close()
            raise

    async def _generate(self, text, voice):
        await self.start()
        self.process.stdin.write((json.dumps({'text': text, 'voice': voice}) + '\n').encode())
        await self.process.stdin.drain()
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), 180)
            if not line:
                raise RuntimeError('Speech worker stopped.')
            result = json.loads(line)
        except BaseException:
            await self.close()
            raise
        if 'error' in result:
            raise ValueError(result['error'])
        audio = base64.b64decode(result['audio'], validate=True)
        if not audio.startswith(b'RIFF'):
            raise RuntimeError('Invalid audio response.')
        return audio, result

    async def speak(self, text, voice):
        key = (text, voice)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key], True
        # No unbounded CPU queue when multiple tabs click at once.
        if self.lock.locked():
            raise SpeechBusy('Speech is busy. Try again in a moment.')
        async with self.lock:
            task = asyncio.create_task(self._generate(text, voice))
            try:
                audio, details = await asyncio.shield(task)
            except asyncio.CancelledError:
                # Consume the response before releasing the lock, keeping the protocol aligned.
                try:
                    await task
                except Exception:
                    pass
                raise
            self.cache[key] = audio
            self.cache_bytes += len(audio)
            while len(self.cache) > 32 or self.cache_bytes > 16 * 1024 * 1024:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= len(old)
            logger.info('CPU speech %s: %.2fs synthesis, %.2fs audio', voice, details['seconds'], details['duration'])
            return audio, False

    async def close(self):
        if self.process and self.process.returncode is None:
            self.process.kill()
            await self.process.wait()
        self.process = None
        if self.log_file:
            self.log_file.close()
            self.log_file = None
