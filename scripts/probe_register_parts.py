"""More warps per output tile, without parallelizing the K reduction."""
parts_lib=ctypes.CDLL(str(ROOT/'artifacts/nf4_register_parts.dll'))
parts_lib.launch_register.argtypes=v2lib.launch_register.argtypes
def parts_product(x,packed,scales,state,warps=4,parts=2):
    rows,cols=state.shape
    out=torch.empty((4,1,rows),device=x.device,dtype=x.dtype)
    rc=parts_lib.launch_register(x.data_ptr(),packed.data_ptr(),scales.data_ptr(),state.code.data_ptr(),out.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream,warps,parts)
    if rc:raise RuntimeError(rc)
    return out
parts_results=[];seen=set()
with torch.inference_mode():
    # Raise clocks through ordinary inference before the microbenchmarks.
    run('parts_warmup')
    for m in engine.model.modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        shape=(m.out_features,m.in_features)
        if shape in seen:continue
        seen.add(shape)
        x=torch.randn((4,1,m.in_features),device='cuda',dtype=torch.bfloat16)
        expected=m(x)
        for warps in (1,2,4,8):
            for parts in (1,2,4,8):
                fn=lambda:parts_product(x,m._register_weights,m._register_scales,m.weight.quant_state,warps,parts)
                row={'shape':shape,'warps':warps,'parts':parts,'identical':torch.equal(expected,fn()),'ms':graph_time(fn)}
                parts_results.append(row)
        print('PARTS_BEST',sorted(parts_results[-16:],key=lambda r:r['ms'])[:4],flush=True)
(ROOT/'artifacts/register-parts-micro.json').write_text(json.dumps(parts_results,indent=2))
print('PARTS_MICRO_DONE',flush=True)
