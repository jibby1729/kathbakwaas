"""Run in interactive_probe after the register graphs have warmed."""
from probe_palette import graph_time
import gc
micro=[];seen=set()
with torch.inference_mode():
    for m in engine.model.modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        shape=(m.out_features,m.in_features)
        if shape in seen:continue
        seen.add(shape)
        x=torch.randn((4,1,m.in_features),device='cuda',dtype=torch.bfloat16)
        weight=bnb.functional.dequantize_4bit(m.weight.t(),m.weight.quant_state).to(m.compute_dtype).t().contiguous()
        expected=m._register_original(x)
        actual=torch.nn.functional.linear(x,weight)
        row={'shape':shape,'identical':torch.equal(expected,actual),
             'register_ms':graph_time(lambda:m(x)),
             'expanded_ms':graph_time(lambda:torch.nn.functional.linear(x,weight)),
             'size_mb':weight.numel()*2/2**20}
        micro.append(row);print(row,flush=True)
        del weight
print('MICRO_DONE',flush=True)
(ROOT/'artifacts/hybrid-micro.json').write_text(json.dumps(micro,indent=2))
