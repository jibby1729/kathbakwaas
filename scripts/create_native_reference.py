"""Save fresh stock-model token sequences for repeatable native-backend checks."""
import json
import os
from pathlib import Path
import sys
from threading import Event

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ['KOSHUR_OPTIMIZE_EXECUTION'] = '0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from app.engine import TranslationEngine

SOURCE = "oh my god you are driving me crazy and you are so annoying and i can't stand you"
TEXTS = [SOURCE, 'How are you?', 'Please close the window.', SOURCE + ' right now',
         "I will arrive tomorrow at 5 o'clock. Please wait for me near the bridge.",
         'My grandmother likes drinking tea in the morning.', 'Please bring two cups of tea.',
         'I cannot come today, but I can come tomorrow.', 'Where is the nearest hospital?',
         'My name is Aamir. I live in Srinagar.', 'Is this yours or mine?',
         'Please do not forget to bring my notebook.',
         'Please tell my brother that I will arrive tomorrow morning. Ask him to wait for me near the bridge, '
         'and bring some tea and bread for our grandmother. If it rains, we can sit inside the house and talk until the rain stops.']

engine = TranslationEngine()
engine.load()
assert engine.state == 'ready', engine.error
original_generate = engine.model.generate
tokens = None


def generate(*args, **kwargs):
    global tokens
    result = original_generate(*args, **kwargs)
    tokens = result.cpu().tolist()
    return result


engine.model.generate = generate
engine.translate('Hello.', Event())
rows = []
directory = ROOT / 'artifacts'
directory.mkdir(exist_ok=True)
for text in TEXTS:
    result = engine.translate(text, Event())
    rows.append({'phase': 'baseline', 'source': text, 'tokens': tokens, **result})
    (directory / 'native-baseline-reference.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print('BASELINE', len(rows), result['timings'], flush=True)
