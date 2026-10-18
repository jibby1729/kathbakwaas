"""Adopt the coalesced scale layout only after exact projection comparisons."""
checks=0;failures=[]
torch.cuda.synchronize()
decoder.graphs.clear();decoder.pool=torch.cuda.graph_pool_handle()
prefills.graphs.clear();prefills.pool=torch.cuda.graph_pool_handle()
with torch.inference_mode():
    for name,m in engine.model.named_modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        repacked=pack_scales(m._register_scales,m.weight.quant_state.shape)
        warps=4 if m.out_features==10240 else 1
        for amplitude in (.1,1,10):
            x=torch.randn((4,1,m.in_features),device='cuda',dtype=torch.bfloat16)*amplitude
            reference=v2product(x,m._register_weights,m._register_scales,m.weight.quant_state,m._register_warps,m._register_unroll)
            actual=scales_product(x,m._register_weights,repacked,m.weight.quant_state,warps,16)
            checks+=1
            if not torch.equal(reference,actual):failures.append(name)
        m._register_scales=repacked;m._register_warps=warps;m._register_unroll=16
assert not failures,failures
product=scales_product
(ROOT/'artifacts/register-scales-real-checks.json').write_text(json.dumps({'checks':checks,'failures':failures},indent=2))
print('SCALES_REAL_CHECKS',checks,flush=True)
gc.collect();torch.cuda.empty_cache()
run('scales_capture')
run('scales_graph')
run('scales_graph_repeat')
print('SCALES_READY',flush=True)
