from grouped_projections import GroupedProjections
torch.cuda.synchronize()
decoder.graphs.clear();decoder.pool=torch.cuda.graph_pool_handle()
prefills.graphs.clear();prefills.pool=torch.cuda.graph_pool_handle()
groups=[];checks=0
with torch.inference_mode():
    for layer in engine.model.model.layers:
        attn=layer.self_attn
        group=GroupedProjections([attn.q_proj,attn.k_proj,attn.v_proj],scales_product)
        for amplitude in (.1,1,10):
            x=torch.randn((4,1,group.in_features),device='cuda',dtype=torch.bfloat16)*amplitude
            expected=[fn(x) for fn in group.originals]
            actual=group.combined(x)
            assert all(torch.equal(a,b) for a,b in zip(expected,actual))
            checks+=len(expected)
        group.install();groups.append(group)
print('GROUPED_REAL_CHECKS',checks,flush=True)
gc.collect();torch.cuda.empty_cache()
run('grouped_capture')
run('grouped_graph')
run('grouped_graph_repeat')
