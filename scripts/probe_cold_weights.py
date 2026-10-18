"""Rotate >64 MiB of weights so a microbenchmark cannot live in GPU L2."""
import math
cold_results=[];seen=set()
with torch.inference_mode():
    run('cold_probe_warmup')
    for m in engine.model.modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        shape=(m.out_features,m.in_features)
        if shape in seen:continue
        seen.add(shape)
        count=max(4,min(32,math.ceil(64*2**20/(m._register_weights.numel()*4))))
        copies=[m._register_weights.clone() for _ in range(count)]
        x=torch.randn((4,1,m.in_features),device='cuda',dtype=torch.bfloat16)
        repacked_scales=pack_scales(m._register_scales,shape)
        for mode in ('v2','parts','scales'):
            for warps in (1,2,4,8):
                for part in ((1,2,4,8) if mode=='parts' else (16,)):
                    fn=(lambda:[parts_product(x,w,m._register_scales,m.weight.quant_state,warps,part) for w in copies]) if mode=='parts' else (
                        (lambda:[scales_product(x,w,repacked_scales,m.weight.quant_state,warps,part) for w in copies]) if mode=='scales' else
                        (lambda:[v2product(x,w,m._register_scales,m.weight.quant_state,warps,part) for w in copies]))
                    row={'shape':shape,'copies':count,'mode':mode,'warps':warps,'part':part,'ms':graph_time(fn)/count}
                    cold_results.append(row)
        print('COLD_BEST',sorted(cold_results[-24:],key=lambda r:r['ms'])[:6],flush=True)
        del copies
        gc.collect();torch.cuda.empty_cache()
(ROOT/'artifacts/cold-weight-micro.json').write_text(json.dumps(cold_results,indent=2))
print('COLD_PROBE_DONE',flush=True)
