"""Benchmark shape-exact whole-decoder capture against unchanged four-beam output."""
import os
import sys
import json
import types
import math
from pathlib import Path
from threading import Event
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['KOSHUR_OPTIMIZE_EXECUTION']='0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import torch
import bitsandbytes as bnb
from app.engine import TranslationEngine
from decode_graph import DecoderGraphs

engine=TranslationEngine()
engine.load()
assert engine.state=='ready',engine.error
source="oh my god you are driving me crazy and you are so annoying and i can't stand you"
texts=[source,'How are you?','Please close the window.',source+' right now']
refs={}
rows=[]
def run(mode,text):
    result=engine.translate(text,Event())
    ref=refs.setdefault(text,(result['draft'],result['translation']))
    row={'mode':mode,'source':text,'identical':ref==(result['draft'],result['translation']),**result}
    rows.append(row)
    (ROOT/'artifacts'/'full-graph-benchmark.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k not in ('source','translation','draft')}),flush=True)
for text in texts: run('baseline',text)
used=0
with torch.inference_mode():
    for m in sorted([m for m in engine.model.modules() if isinstance(m,bnb.nn.Linear4bit)],key=lambda m:math.prod(m.weight.quant_state.shape)):
        size=math.prod(m.weight.quant_state.shape)*2
        if used+size>2300*2**20: break
        m._expanded_nf4=bnb.functional.dequantize_4bit(m.weight.t(),m.weight.quant_state).to(m.compute_dtype).t()
        def forward(self,x):
            return torch.nn.functional.linear(x.to(self.compute_dtype),self._expanded_nf4,self.bias).to(x.dtype)
        m.forward=types.MethodType(forward,m)
        used+=size
decoder=DecoderGraphs(engine.model)
for text in texts:
    run('capture',text)
    run('graph_replay',text)
    run('graph_replay',text)
print('DONE',flush=True)
