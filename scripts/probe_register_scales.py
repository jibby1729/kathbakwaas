"""Coalesce scale reads without changing any scale values."""
scales_lib=ctypes.CDLL(str(ROOT/'artifacts/nf4_register_scales.dll'))
scales_lib.launch_register.argtypes=v2lib.launch_register.argtypes
def pack_scales(scales,shape):
    rows,cols=shape
    return scales.reshape(rows//16,16,cols//64).transpose(1,2).contiguous().flatten()
def scales_product(x,packed,scales,state,warps=1,unroll=8):
    rows,cols=state.shape
    out=torch.empty((4,1,rows),device=x.device,dtype=x.dtype)
    rc=scales_lib.launch_register(x.data_ptr(),packed.data_ptr(),scales.data_ptr(),state.code.data_ptr(),out.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream,warps,unroll)
    if rc:raise RuntimeError(rc)
    return out
scales_results=[];seen=set()
with torch.inference_mode():
    run('scales_warmup')
    for m in engine.model.modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        shape=(m.out_features,m.in_features)
        if shape in seen:continue
        seen.add(shape)
        x=torch.randn((4,1,m.in_features),device='cuda',dtype=torch.bfloat16)
        expected=m(x);repacked_scales=pack_scales(m._register_scales,shape)
        for warps in (1,2,4,8):
            for unroll in (1,4,8,16):
                fn=lambda:scales_product(x,m._register_weights,repacked_scales,m.weight.quant_state,warps,unroll)
                row={'shape':shape,'warps':warps,'unroll':unroll,'identical':torch.equal(expected,fn()),'ms':graph_time(fn)}
                scales_results.append(row)
        print('SCALES_BEST',sorted(scales_results[-16:],key=lambda r:r['ms'])[:4],flush=True)
(ROOT/'artifacts/register-scales-micro.json').write_text(json.dumps(scales_results,indent=2))
print('SCALES_MICRO_DONE',flush=True)
