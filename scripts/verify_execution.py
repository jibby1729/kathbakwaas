"""Fresh-request baseline/optimized token identity, cancellation and prefix checks."""
import os
import sys
import json
from pathlib import Path
from threading import Event
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['KOSHUR_OPTIMIZE_EXECUTION']='0'
import torch
from app.engine import TranslationEngine,TranslationCancelled
from app.execution import optimize_execution

engine=TranslationEngine()
engine.load()
assert engine.state=='ready',engine.error
source="oh my god you are driving me crazy and you are so annoying and i can't stand you"
texts=[source,'How are you?','Please close the window.',source+' right now',
       'I will arrive tomorrow at 5 o\'clock. Please wait for me near the bridge.',
       'Do not open the door. It is cold outside.',
       'My grandmother likes drinking tea in the morning.',
       'Where can I buy two kilograms of apples?',
       'The bus was late, so we walked home together in the rain.',
       'Thank you for helping me. I am feeling much better now.']
original_generate=engine.model.generate
tokens=None
def generate(*args,**kwargs):
    global tokens
    result=original_generate(*args,**kwargs)
    tokens=result.cpu().tolist()
    return result
engine.model.generate=generate
refs={};rows=[]
def run(mode,text,session=None):
    result=engine.translate(text,Event(),session=session)
    ref=refs.setdefault(text,(tokens,result['draft'],result['translation']))
    row={'mode':mode,'source':text,'identical_tokens':ref[0]==tokens,
         'identical_text':ref[1:]==(result['draft'],result['translation']),**result}
    rows.append(row)
    (ROOT/'artifacts'/'execution-verification.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k not in ('source','translation','draft')}),flush=True)
engine.translate('Hello.',Event())
for text in texts:run('baseline',text)
setup=optimize_execution(engine.model)
print('SETUP '+json.dumps(setup),flush=True)
assert setup['mlp_graphs']==34,setup
for text in texts:run('optimized',text)
for _ in range(2):run('optimized_repeat',source)
engine.prefill('oh', 'verify')
engine.prefill('oh my', 'verify')
engine.prefill(source,'verify')
run('optimized_prefix',source,'verify')
cancelled=Event();cancelled.set()
try:
    engine.translate(source,cancelled)
    raise AssertionError('Cancellation failed')
except TranslationCancelled:pass
run('after_cancel',source)
assert all(r['identical_tokens'] and r['identical_text'] for r in rows)
print('ALL_IDENTICAL; DONE',flush=True)
