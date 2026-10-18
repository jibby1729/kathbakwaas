"""Persistent Bolbosh CPU worker. JSON lines in; base64 WAV responses out."""
import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor' / 'bolbosh'))
SAMPLE_RATE = 22050
VOICES = {'male': 422, 'female': 423}


def split_text(text, limit=180):
    """Bound each model call without dropping text or splitting combining marks."""
    parts = []
    for sentence in re.split(r'(?<=[۔.!?؟])\s*|\n+', text):
        current = ''
        for word in sentence.split():
            if len(word) > limit:
                raise ValueError('Please use shorter words or sentences for speech.')
            if current and len(current) + len(word) + 1 > limit:
                parts.append(current)
                current = ''
            current = f'{current} {word}'.strip()
        if current:
            parts.append(current)
    return parts


class Synthesizer:
    def __init__(self):
        import torch
        from matcha.models.matcha_tts import MatchaTTS
        from matcha.hifigan.models import Generator
        from matcha.hifigan.config import v1
        from matcha.hifigan.env import AttrDict
        self.torch = torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        # This worker requires the separate CPU build, never the translator's CUDA build.
        if torch.version.cuda is not None:
            raise RuntimeError('Install the CPU-only runtime from requirements-tts.txt.')
        weights = ROOT / '.model-cache' / 'bolbosh'
        # Official, pinned Lightning checkpoint contains OmegaConf metadata.
        checkpoint = torch.load(weights / 'model.ckpt', map_location='cpu', weights_only=False)
        import inspect
        params = inspect.signature(MatchaTTS.__init__).parameters
        config = {k: v for k, v in checkpoint['hyper_parameters'].items() if k in params}
        self.model = MatchaTTS(**config).cpu().float().eval()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        del checkpoint
        self.vocoder = Generator(AttrDict(v1)).cpu().float().eval()
        self.vocoder.load_state_dict(torch.load(weights / 'generator_v1', map_location='cpu', weights_only=True)['generator'])
        self.vocoder.remove_weight_norm()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.vocoder_parameters = sum(p.numel() for p in self.vocoder.parameters())

    def speak(self, text, voice):
        import numpy as np
        from matcha.text import text_to_sequence
        torch = self.torch
        chunks = []
        with torch.inference_mode():
            for part in split_text(text):
                part = part.replace('ي', 'ی').replace('ك', 'ک').replace('۔', '').strip()
                sequence, _ = text_to_sequence(part, ['basic_cleaners'])
                if not sequence:
                    raise ValueError('There is no speakable Kashmiri text.')
                interspersed = [0] * (len(sequence) * 2 + 1)
                interspersed[1::2] = sequence
                x = torch.tensor([interspersed], dtype=torch.long, device='cpu')
                lengths = torch.tensor([x.shape[-1]], device='cpu')
                speaker = torch.tensor([VOICES[voice]], dtype=torch.long, device='cpu')
                out = self.model.synthesise(x, lengths, n_timesteps=10, temperature=0.667,
                                            spks=speaker, length_scale=1.0)
                audio = self.vocoder(out['mel']).clamp(-1, 1).squeeze().numpy()
                if not np.isfinite(audio).all():
                    raise RuntimeError('Speech synthesis produced invalid audio.')
                if chunks:
                    chunks.append(np.zeros(int(SAMPLE_RATE * .18), dtype=np.float32))
                chunks.append(audio)
                if sum(len(c) for c in chunks) > SAMPLE_RATE * 180:
                    raise ValueError('Speech is too long. Try a shorter passage.')
        if not chunks:
            raise ValueError('There is no speakable Kashmiri text.')
        audio = np.concatenate(chunks)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes((audio * 32767).astype('<i2').tobytes())
        return buffer.getvalue(), len(audio) / SAMPLE_RATE


def main():
    # Upstream startup messages must not interfere with the JSON protocol.
    with redirect_stdout(sys.stderr):
        synth = Synthesizer()
    print(json.dumps({'ready': True, 'device': 'cpu', 'parameters': synth.parameters,
                      'vocoder_parameters': synth.vocoder_parameters}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            text, voice = request['text'], request['voice']
            if voice not in VOICES or not isinstance(text, str) or not 1 <= len(text.strip()) <= 1500:
                raise ValueError('Invalid speech request.')
            start = time.perf_counter()
            with redirect_stdout(sys.stderr):
                wav, duration = synth.speak(text, voice)
            result = {'audio': base64.b64encode(wav).decode('ascii'), 'duration': duration,
                      'seconds': time.perf_counter() - start}
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            result = {'error': str(exc)}
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
