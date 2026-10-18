"""Verify the app's native backend against saved full original token sequences."""
import os,sys,json,time,logging
from pathlib import Path
from threading import Event,Timer
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['KOSHUR_OPTIMIZE_EXECUTION']='1'
logging.basicConfig(level=logging.INFO)
import torch
from app.engine import TranslationEngine,TranslationCancelled
reference_file=ROOT/'artifacts/native-baseline-reference.json'
if not reference_file.exists():reference_file=ROOT/'artifacts/full-pipeline-checks.json'
if not reference_file.exists():raise RuntimeError('Run scripts/create_native_reference.py first')
references={row['source']:row for row in json.loads(reference_file.read_text(encoding='utf-8')) if row['phase']=='baseline'}
engine=TranslationEngine();started=time.perf_counter();engine.load()
assert engine.state=='ready',engine.error
assert engine.execution_optimization.get('backend')=='native_nf4',engine.execution_optimization
print('NATIVE_READY',round(time.perf_counter()-started,2),engine.execution_optimization,flush=True)
original_generate=engine.model.generate;tokens=None;rows=[]
report=ROOT/'artifacts'/os.getenv('KOSHUR_VERIFY_REPORT','native-backend-verification.json')
def generate(*args,**kwargs):
    global tokens
    result=original_generate(*args,**kwargs);tokens=result.cpu().tolist();return result
engine.model.generate=generate
def run(mode,text,session=None):
    result=engine.translate(text,Event(),session)
    reference=references[text]
    row={'mode':mode,'source':text,'identical_tokens':tokens==reference['tokens'],
         'identical_text':result['draft']==reference['draft'] and result['translation']==reference['translation'],
         'allocated_mb':torch.cuda.memory_allocated()/2**20,'reserved_mb':torch.cuda.memory_reserved()/2**20,**result}
    rows.append(row)
    report.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k not in ('source','translation','draft')}),flush=True)
    assert row['identical_tokens'] and row['identical_text'],text
for text in references:
    run('first_request',text)
    run('warm_request',text)
source=next(iter(references))
for _ in range(3):run('repeat',source)
engine.prefill('oh my','native-verify')
engine.prefill('oh my goodness','native-verify')
engine.prefill(source,'native-verify')
run('typed_prefix',source,'native-verify')
cancelled=Event();timer=Timer(.15,cancelled.set);timer.start()
try:
    engine.translate(list(references)[-1],cancelled)
    raise AssertionError('In-flight cancellation was not observed')
except TranslationCancelled:
    print('IN_FLIGHT_CANCELLED',flush=True)
finally:timer.cancel()
run('after_cancel',source)
assert engine.fast_execution['decoder'].fallback_count>0
print('NATIVE_VERIFIED',len(references),'fallbacks',engine.fast_execution['decoder'].fallback_count,flush=True)
