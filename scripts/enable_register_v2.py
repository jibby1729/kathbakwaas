"""Replace experimental graphs after validating every real linear projection."""
checks=0;failures=[]
with torch.inference_mode():
    for name,m in engine.model.named_modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        m._register_warps=1
        m._register_unroll=16 if m.in_features==10240 else 8
        for amplitude in (.1,1,10):
            x=torch.randn((4,1,m.in_features),device='cuda',dtype=torch.bfloat16)*amplitude
            reference=m._register_original(x)
            actual=v2product(x,m._register_weights,m._register_scales,m.weight.quant_state,m._register_warps,m._register_unroll)
            checks+=1
            if not torch.equal(reference,actual):failures.append({'name':name,'amplitude':amplitude,'different':int((reference!=actual).sum())})
print('V2_REAL_CHECKS',checks,'FAILURES',failures,flush=True)
(ROOT/'artifacts/register-v2-real-checks.json').write_text(json.dumps({'checks':checks,'failures':failures},indent=2))
assert not failures
torch.cuda.synchronize()
decoder.graphs.clear();decoder.pool=torch.cuda.graph_pool_handle()
product=v2product
gc.collect();torch.cuda.empty_cache()
run('v2_capture')
run('v2_graph')
run('v2_graph_repeat')
print('V2_READY',flush=True)
