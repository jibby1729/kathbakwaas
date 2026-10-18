"""Test a lossless quantized layout permutation and direct tensor-core decoding."""
import ctypes,json,itertools
from pathlib import Path
import torch
import bitsandbytes as bnb
from probe_palette import graph_time
ROOT=Path(__file__).resolve().parent.parent
lib=ctypes.CDLL(str(ROOT/'artifacts/nf4_register.dll'))
lib.pack_nf4.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p]
lib.launch_register.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_void_p]+[ctypes.c_int]*2

def prepare(w,state):
    rows,cols=state.shape
    packed=torch.empty((rows*cols//8,),device=w.device,dtype=torch.int32)
    rc=lib.pack_nf4(w.data_ptr(),packed.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream)
    if rc:raise RuntimeError(rc)
    scales=bnb.functional.dequantize_blockwise(state.absmax,state.state2)+state.offset
    return packed,scales

def product(x,packed,scales,state,warps=4,unroll=4):
    rows,cols=state.shape
    out=torch.empty((4,1,rows),device=x.device,dtype=x.dtype)
    rc=lib.launch_register(x.data_ptr(),packed.data_ptr(),scales.data_ptr(),state.code.data_ptr(),out.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream,warps,unroll)
    if rc:raise RuntimeError(rc)
    return out

if __name__=='__main__':
    records=[];torch.manual_seed(1729)
    with torch.inference_mode():
        for rows,cols in [(1024,2560),(2048,2560),(2560,2048),(10240,2560),(2560,10240)]:
            dense=torch.randn((rows,cols),device='cuda',dtype=torch.bfloat16)*.02
            w,state=bnb.functional.quantize_4bit(dense,compress_statistics=True,quant_type='nf4')
            packed,scales=prepare(w,state)
            x=torch.randn((4,1,cols),device='cuda',dtype=torch.bfloat16)
            ref=bnb.matmul_4bit(x,w.t(),quant_state=state)
            row={'shape':[rows,cols],'mode':'baseline','ms':graph_time(lambda:bnb.matmul_4bit(x,w.t(),quant_state=state))}
            records.append(row);print(json.dumps(row),flush=True)
            for warps,unroll in itertools.product([1,2,4,8],[1,4,8,16]):
                fn=lambda:product(x,packed,scales,state,warps,unroll)
                out=fn()
                row={'shape':[rows,cols],'mode':'register','warps':warps,'unroll':unroll,'equal':torch.equal(ref,out),
                     'different':int((ref!=out).sum()),'max_abs':float((ref.float()-out.float()).abs().max()),'ms':graph_time(fn)}
                records.append(row)
            print('BEST '+json.dumps(sorted([r for r in records if r['shape']==[rows,cols]],key=lambda r:r['ms'])[:5]),flush=True)
            (ROOT/'artifacts/register-probe.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
