"""Execution-only experiments. No production changes; no decoder reductions."""
import os
import sys
import json
import types
import gc
from pathlib import Path
from threading import Event
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['KOSHUR_OPTIMIZE_EXECUTION'] = '0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import torch
import bitsandbytes as bnb
from app.engine import TranslationEngine

engine = TranslationEngine()
engine.load()
assert engine.state == 'ready', engine.error
source = "oh my god you are driving me crazy and you are so annoying and i can't stand you"
report = []
references = {}

def trial(label, texts=None, repeats=2):
    texts = texts or [source]
    for text in texts:
        for repeat in range(repeats):
            torch.cuda.synchronize()
            start = perf_counter()
            result = engine.translate(text, Event())
            torch.cuda.synchronize()
            elapsed = (perf_counter()-start)*1000
            ref = references.setdefault(text, (result['draft'], result['translation']))
            row = {'mode': label, 'source': text, 'repeat': repeat,
                   'wall_ms': elapsed, 'identical': ref == (result['draft'],result['translation']),
                   'memory_mb': torch.cuda.memory_allocated()/2**20, **result}
            report.append(row)
            (ROOT/'artifacts'/'execution-benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k not in ('source','translation','draft')},ensure_ascii=True),flush=True)

texts = [source, 'How are you?', 'Please close the window.', source+' right now']
trial('baseline', texts, 1)
for count in (1,2,4):
    torch.set_num_threads(count)
    trial('threads_'+str(count))

# Cache precisely the same dequantization and transpose used by MatMul4Bit.
# Sort by size to eliminate the most calls within a bounded memory budget.
linears = sorted([(name,m) for name,m in engine.model.named_modules() if isinstance(m,bnb.nn.Linear4bit)],
                 key=lambda pair: pair[1].weight.quant_state.shape.numel() if hasattr(pair[1].weight.quant_state.shape,'numel') else __import__('math').prod(pair[1].weight.quant_state.shape))
used = 0
originals = []
with torch.inference_mode():
    for name,m in linears:
        size = __import__('math').prod(m.weight.quant_state.shape)*2
        if used + size > 2300*2**20:
            break
        m._expanded_nf4 = bnb.functional.dequantize_4bit(m.weight.t(),m.weight.quant_state).to(m.compute_dtype).t()
        original = m.forward
        originals.append((m,original))
        def forward(self,x):
            return torch.nn.functional.linear(x.to(self.compute_dtype),self._expanded_nf4,self.bias).to(x.dtype)
        m.forward = types.MethodType(forward,m)
        used += size
print(json.dumps({'cached_linears':len(originals),'expanded_mb':used/2**20}),flush=True)
trial('cached_dequantization',texts,2)

class GraphMLP:
    def __init__(self,module,pool):
        self.original = module.forward
        self.x = torch.zeros((4,1,engine.model.config.hidden_size),device='cuda',dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                self.original(self.x)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,pool=pool), torch.inference_mode():
            self.y = self.original(self.x)
    def __call__(self,x):
        if tuple(x.shape) != tuple(self.x.shape):
            return self.original(x)
        self.x.copy_(x)
        self.graph.replay()
        return self.y

pool = torch.cuda.graph_pool_handle()
graphs = []
for layer in engine.model.model.layers:
    graph = GraphMLP(layer.mlp,pool)
    layer.mlp.forward = graph
    graphs.append(graph)
print('MLP graphs ready',flush=True)
trial('cached_dequantization_mlp_graphs',texts,2)
for layer,g in zip(engine.model.model.layers,graphs):
    layer.mlp.forward = g.original
graphs.clear()
del graph,g
gc.collect()
torch.cuda.empty_cache()
for m,original in originals:
    m.forward = original
    del m._expanded_nf4
torch.cuda.empty_cache()
print('DONE',flush=True)
