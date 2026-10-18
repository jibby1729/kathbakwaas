"""Fresh translations with losslessly packed NF4 and direct tensor-core kernels."""
import os,sys,json,types,time
from pathlib import Path
from threading import Event
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['KOSHUR_OPTIMIZE_EXECUTION']='0'
import torch
import bitsandbytes as bnb
from app.engine import TranslationEngine
from decode_graph import DecoderGraphs
from probe_register import prepare,product

engine=TranslationEngine();engine.load()
assert engine.state=='ready',engine.error
source="oh my god you are driving me crazy and you are so annoying and i can't stand you"
texts=[source,'How are you?','Please close the window.',source+' right now',
       "I will arrive tomorrow at 5 o'clock. Please wait for me near the bridge.",
       'My grandmother likes drinking tea in the morning.']
refs={};records=[];tokens=None
original_generate=engine.model.generate
def generate(*args,**kwargs):
    global tokens
    result=original_generate(*args,**kwargs);tokens=result.cpu().tolist();return result
engine.model.generate=generate
def run(mode,text):
    result=engine.translate(text,Event())
    ref=refs.setdefault(text,(tokens,result['draft'],result['translation']))
    row={'mode':mode,'source':text,'identical_tokens':ref[0]==tokens,
         'identical_text':ref[1:]==(result['draft'],result['translation']),
         'allocated_mb':torch.cuda.memory_allocated()/2**20,'reserved_mb':torch.cuda.memory_reserved()/2**20,**result}
    records.append(row)
    (ROOT/'artifacts/register-model.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k not in ('source','draft','translation')}),flush=True)
engine.translate('Hello.',Event())
for text in texts:run('baseline',text)
checks=[];torch.manual_seed(1729)
with torch.inference_mode():
    for name,m in engine.model.named_modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        original=m.forward;state=m.weight.quant_state
        packed,scales=prepare(m.weight,state)
        warps,unroll=(2,4) if m.out_features==10240 else ((1,8) if m.in_features==10240 else (1,16))
        for scale in (.1,1,10):
            x=torch.randn((4,1,m.in_features),device='cuda',dtype=torch.bfloat16)*scale
            a=original(x);b=product(x,packed,scales,state,warps,unroll)
            check={'name':name,'scale':scale,'equal':torch.equal(a,b),'different':int((a!=b).sum())}
            checks.append(check)
            if not check['equal']: print('MISMATCH '+json.dumps(check),flush=True)
        m._register_original=original;m._register_weights=packed;m._register_scales=scales
        m._register_warps=warps;m._register_unroll=unroll
        def forward(self,x):
            if x.shape!=(4,1,self.in_features):return self._register_original(x)
            return product(x,self._register_weights,self._register_scales,self.weight.quant_state,self._register_warps,self._register_unroll)
        m.forward=types.MethodType(forward,m)
    del a,b,x,packed,scales
    torch.cuda.empty_cache()
(ROOT/'artifacts/register-real-checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
assert all(r['equal'] for r in checks),'Numerical gate failed'
print('ALL_REAL_MATRICES_EQUAL '+str(len(checks)),flush=True)
run('register_eager',source)
decoder=DecoderGraphs(engine.model)
for text in texts:
    run('capture',text)
    run('register_graph',text)
    run('register_graph',text)
assert all(r['identical_tokens'] and r['identical_text'] for r in records),'Output regression'
print('ALL_IDENTICAL; DONE',flush=True)
