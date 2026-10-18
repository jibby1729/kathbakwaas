"""Model-level numerical gate and timing for the fused tensor-core experiment."""
import os
import sys
import json
import types
from pathlib import Path
from threading import Event
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['KOSHUR_OPTIMIZE_EXECUTION']='0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import torch
import bitsandbytes as bnb
from app.engine import TranslationEngine
from decode_graph import DecoderGraphs
from probe_nf4_kernel import wmma

engine=TranslationEngine()
engine.load()
assert engine.state=='ready',engine.error
source="oh my god you are driving me crazy and you are so annoying and i can't stand you"
texts=[source,'How are you?','Please close the window.',source+' right now']
refs={};rows=[]
def run(mode,text):
    result=engine.translate(text,Event())
    ref=refs.setdefault(text,(result['draft'],result['translation']))
    row={'mode':mode,'source':text,'identical':ref==(result['draft'],result['translation']),**result}
    rows.append(row)
    (ROOT/'artifacts'/'fused-graph-benchmark.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k not in ('source','translation','draft')}),flush=True)
for text in texts:run('baseline',text)
checks=[]
torch.manual_seed(1729)
with torch.inference_mode():
    for name,m in engine.model.named_modules():
        if not isinstance(m,bnb.nn.Linear4bit): continue
        original=m.forward
        state=m.weight.quant_state
        scales=bnb.functional.dequantize_blockwise(state.absmax,state.state2)+state.offset
        group=64 if m.out_features==10240 else 256
        # Gate EVERY real matrix before replacing it, using several activation scales.
        for scale in (.1,1,10):
            x=torch.randn((4,1,m.in_features),device='cuda',dtype=torch.bfloat16)*scale
            a=original(x)
            b=wmma(x,m.weight,state,scales,0,group)
            check={'name':name,'scale':scale,'equal':torch.equal(a,b),'different':int((a!=b).sum())}
            checks.append(check)
            if not check['equal']:
                print('NUMERICAL_MISMATCH '+json.dumps(check),flush=True)
        m._original_forward=original
        m._exact_scales=scales
        m._kernel_group=group
        # Dense cache for down projection and attention; fused reads for the big up/gate projections.
        if m.out_features!=10240:
            m._expanded_nf4=bnb.functional.dequantize_4bit(m.weight.t(),state).to(m.compute_dtype).t()
        def forward(self,x):
            if x.shape[:2]!=(4,1):return self._original_forward(x)
            if hasattr(self,'_expanded_nf4'):
                return torch.nn.functional.linear(x.to(self.compute_dtype),self._expanded_nf4,self.bias).to(x.dtype)
            return wmma(x,self.weight,self.weight.quant_state,self._exact_scales,0,self._kernel_group)
        m.forward=types.MethodType(forward,m)
(ROOT/'artifacts'/'fused-real-weight-checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('REAL_MATRICES_EQUAL '+str(all(r['equal'] for r in checks)),flush=True)
if not all(r['equal'] for r in checks):
    raise RuntimeError('Fused kernel rejected by numerical gate')
decoder=DecoderGraphs(engine.model)
for text in texts:
    run('capture',text)
    run('fused_graph',text)
    run('fused_graph',text)
print('DONE',flush=True)
