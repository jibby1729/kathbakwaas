"""Compare preexpanded codebooks while preserving BF16 operand values."""
import probe_palette
palette_lib=ctypes.CDLL(str(ROOT/'artifacts/nf4_register_palette.dll'))
palette_lib.launch_register.argtypes=v2lib.launch_register.argtypes
def palette_product(x,packed,palette,state,warps=1,unroll=16):
    rows,cols=state.shape
    out=torch.empty((4,1,rows),device=x.device,dtype=x.dtype)
    rc=palette_lib.launch_register(x.data_ptr(),packed.data_ptr(),palette.data_ptr(),state.code.data_ptr(),out.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream,warps,unroll)
    if rc:raise RuntimeError(rc)
    return out
palette_results=[];seen=set()
with torch.inference_mode():
    for m in engine.model.modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        shape=(m.out_features,m.in_features)
        if shape in seen:continue
        seen.add(shape)
        x=torch.randn((4,1,m.in_features),device='cuda',dtype=torch.bfloat16)
        expected=m(x)
        _,palette=probe_palette.palette_for(m.weight,m.weight.quant_state)
        for warps in (1,2,4):
            for unroll in (4,8,16):
                fn=lambda:palette_product(x,m._register_weights,palette,m.weight.quant_state,warps,unroll)
                row={'shape':shape,'warps':warps,'unroll':unroll,'identical':torch.equal(expected,fn()),'ms':graph_time(fn)}
                palette_results.append(row)
        print('PALETTE_BEST',sorted(palette_results[-9:],key=lambda r:r['ms'])[:3],flush=True)
        del palette
(ROOT/'artifacts/register-palette-micro.json').write_text(json.dumps(palette_results,indent=2))
print('PALETTE_MICRO_DONE',flush=True)
