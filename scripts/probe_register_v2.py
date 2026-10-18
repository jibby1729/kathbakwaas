"""Compare register shuffles with the original codebook loads."""
import ctypes
v2lib=ctypes.CDLL(str(ROOT/'artifacts/nf4_register_v2.dll'))
v2lib.launch_register.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_void_p]+[ctypes.c_int]*2
def v2product(x,packed,scales,state,warps=1,unroll=16):
    rows,cols=state.shape
    out=torch.empty((4,1,rows),device=x.device,dtype=x.dtype)
    rc=v2lib.launch_register(x.data_ptr(),packed.data_ptr(),scales.data_ptr(),state.code.data_ptr(),out.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream,warps,unroll)
    if rc:raise RuntimeError(rc)
    return out
v2results=[];seen=set()
with torch.inference_mode():
    for m in engine.model.modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        shape=(m.out_features,m.in_features)
        if shape in seen:continue
        seen.add(shape)
        x=torch.randn((4,1,m.in_features),device='cuda',dtype=torch.bfloat16)
        expected=m._register_original(x)
        for warps in (1,2,4):
            for unroll in (4,8,16):
                fn=lambda:v2product(x,m._register_weights,m._register_scales,m.weight.quant_state,warps,unroll)
                row={'shape':shape,'warps':warps,'unroll':unroll,'identical':torch.equal(expected,fn()),'ms':graph_time(fn)}
                v2results.append(row)
        best=sorted(v2results[-9:],key=lambda r:r['ms'])[:3]
        print('V2_BEST',best,flush=True)
(ROOT/'artifacts/register-v2-micro.json').write_text(json.dumps(v2results,indent=2))
print('V2_MICRO_DONE',flush=True)
