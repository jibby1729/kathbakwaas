"""Fetch pinned Bolbosh inference sources and weights, without training dependencies."""
import concurrent.futures
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'bolbosh'
SOURCE = '7435b77029a691cc00a335356013ca6ee78f794a'
NORMALIZER = '9bb966e76b88caf875bfb9facf3036bef706427e'
MODEL = '718203e01e0b4ddab3e925445619c69cddeb7231'


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    temporary = dest.with_suffix(dest.suffix + '.download')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(dest)


def main():
    files = ['LICENSE', 'matcha/__init__.py', 'matcha/models/__init__.py',
             'matcha/models/components/__init__.py', 'matcha/models/matcha_tts.py',
             'matcha/models/components/text_encoder.py', 'matcha/models/components/decoder.py',
             'matcha/models/components/flow_matching.py', 'matcha/models/components/transformer.py',
             'matcha/utils/model.py', 'matcha/text/__init__.py', 'matcha/text/symbols.py',
             'matcha/hifigan/__init__.py', 'matcha/hifigan/models.py', 'matcha/hifigan/config.py',
             'matcha/hifigan/env.py', 'matcha/hifigan/xutils.py', 'matcha/hifigan/LICENSE']
    jobs = [(f'https://raw.githubusercontent.com/gaash-lab/Bolbosh/{SOURCE}/{p}', VENDOR / p) for p in files]
    for p in ['__init__.py', 'constants.py', 'normalizer.py']:
        jobs.append((f'https://raw.githubusercontent.com/abdulmuizz0903/KashmiriNormalizer/{NORMALIZER}/src/KashmiriNormalizer/{p}', VENDOR / 'KashmiriNormalizer' / p))
    jobs.append((f'https://raw.githubusercontent.com/abdulmuizz0903/KashmiriNormalizer/{NORMALIZER}/LICENSE', VENDOR / 'KashmiriNormalizer' / 'LICENSE'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda job: fetch(*job), jobs))
    # Preserve upstream synthesis verbatim; remove training-only paths/imports.
    model = VENDOR / 'matcha/models/matcha_tts.py'
    source = model.read_text(encoding='utf-8')
    source = source.split('    def forward(self, x, x_lengths, y, y_lengths,')[0]
    source = source.replace('import matcha.utils.monotonic_align as monotonic_align  # pylint: disable=consider-using-from-import\n', '')
    source = source.replace('from matcha.models.baselightningmodule import BaseLightningClass', 'from matcha.models.inference_base import BaseLightningClass')
    if 'y_max_length_ > 2600' not in source:
        source = source.replace('        y_max_length_ = fix_len_compatibility(y_max_length)\n',
            '        y_max_length_ = fix_len_compatibility(y_max_length)\n'
            '        # Bound CPU memory before allocating alignment/decoder tensors.\n'
            '        if y_max_length_ > 2600:\n'
            '            raise ValueError("Speech segment is too long. Try a shorter sentence.")\n')
    model.write_text(source, encoding='utf-8')
    (VENDOR / 'matcha/utils/__init__.py').write_text('from logging import getLogger as get_pylogger\n', encoding='utf-8')
    xutils = VENDOR / 'matcha/hifigan/xutils.py'
    source = xutils.read_text(encoding='utf-8')
    if 'import matplotlib\n' in source:
        source = source.replace('import matplotlib\n', '').replace('matplotlib.use("Agg")\n', '')
        source = source.replace('import matplotlib.pylab as plt\n', '')
        source = source.replace('def plot_spectrogram(spectrogram):\n',
                                'def plot_spectrogram(spectrogram):\n    import matplotlib.pylab as plt\n')
        xutils.write_text(source, encoding='utf-8')
    weights = ROOT / '.model-cache' / 'bolbosh'
    for url, path in [
        (f'https://huggingface.co/GAASH-Lab/Matcha-TTS-Kashmiri/resolve/{MODEL}/model.ckpt', weights / 'model.ckpt'),
        ('https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1', weights / 'generator_v1'),
    ]:
        print(f'Downloading {path.name}', flush=True)
        fetch(url, path)
    (weights / 'revisions.json').write_text(json.dumps({'source': SOURCE, 'normalizer': NORMALIZER, 'model': MODEL}, indent=2))
    print('Bolbosh files ready.', flush=True)


if __name__ == '__main__':
    main()
