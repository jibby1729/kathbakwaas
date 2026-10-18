"""Load once for a series of isolated probes: python -i scripts/interactive_probe.py."""
import os,sys,json,types,time
from pathlib import Path
from threading import Event
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
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
refs={};records=[];tokens=None
original_generate=engine.model.generate
def generate(*args,**kwargs):
    global tokens
    result=original_generate(*args,**kwargs);tokens=result.cpu().tolist();return result
engine.model.generate=generate
def run(mode,text=source,session=None):
    result=engine.translate(text,Event(),session)
    ref=refs.setdefault(text,(tokens,result['draft'],result['translation']))
    row={'mode':mode,'source':text,'identical_tokens':ref[0]==tokens,
         'identical_text':ref[1:]==(result['draft'],result['translation']),
         'allocated_mb':torch.cuda.memory_allocated()/2**20,'reserved_mb':torch.cuda.memory_reserved()/2**20,**result}
    records.append(row)
    (ROOT/'artifacts/interactive-probes.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k not in ('source','draft','translation')}),flush=True)
    return row
engine.translate('Hello.',Event())
run('baseline')
with torch.inference_mode():
    for m in engine.model.modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        packed,scales=prepare(m.weight,m.weight.quant_state)
        m._register_original=m.forward;m._register_weights=packed;m._register_scales=scales
        m._register_warps,m._register_unroll=(2,4) if m.out_features==10240 else ((1,8) if m.in_features==10240 else (1,16))
        def forward(self,x):
            if x.shape!=(4,1,self.in_features):return self._register_original(x)
            return product(x,self._register_weights,self._register_scales,self.weight.quant_state,self._register_warps,self._register_unroll)
        m.forward=types.MethodType(forward,m)
    torch.cuda.empty_cache()
decoder=DecoderGraphs(engine.model)
run('capture')
run('register_graph')
print('READY_FOR_PROBES',flush=True)
