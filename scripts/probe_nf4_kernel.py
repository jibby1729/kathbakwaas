"""Numerical and timing gate for an experimental four-vector NF4 kernel."""
import ctypes
import json
from pathlib import Path
import torch
import bitsandbytes as bnb
ROOT=Path(__file__).resolve().parent.parent
lib=ctypes.CDLL(str(ROOT/'artifacts'/'nf4_four_beam.dll'))
launch=lib.launch_nf4_four_beam
launch.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_void_p]
launch.restype=ctypes.c_int
wmma_launch=lib.launch_nf4_wmma
wmma_launch.argtypes=launch.argtypes+[ctypes.c_int]*2
wmma_launch.restype=ctypes.c_int

def fused(x,weight,state,scales):
    rows,cols=state.shape
    y=torch.empty((*x.shape[:-1],rows),device=x.device,dtype=x.dtype)
    status=launch(x.data_ptr(),weight.data_ptr(),scales.data_ptr(),state.code.data_ptr(),y.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream)
    if status: raise RuntimeError(f'CUDA launch failed: {status}')
    return y

def wmma(x,weight,state,scales,splits,group):
    rows,cols=state.shape
    y=torch.empty((*x.shape[:-1],rows),device=x.device,dtype=x.dtype)
    status=wmma_launch(x.data_ptr(),weight.data_ptr(),scales.data_ptr(),state.code.data_ptr(),y.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream,splits,group)
    if status: raise RuntimeError(f'CUDA launch failed: {status}')
    return y

if __name__=='__main__':
    results=[]
    torch.manual_seed(1729)
    with torch.inference_mode():
        for rows,cols in [(1024,2560),(2048,2560),(10240,2560),(2560,10240)]:
            dense=torch.randn((rows,cols),device='cuda',dtype=torch.bfloat16)*.02
            weight,state=bnb.functional.quantize_4bit(dense,compress_statistics=True,quant_type='nf4')
            scales=bnb.functional.dequantize_blockwise(state.absmax,state.state2)+state.offset
            x=torch.randn((4,1,cols),device='cuda',dtype=torch.bfloat16)
            a=bnb.matmul_4bit(x,weight.t(),quant_state=state)
            b=fused(x,weight,state,scales)
            def time(fn):
                for _ in range(3):fn()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):fn()
                end.record();end.synchronize()
                return start.elapsed_time(end)/20
            row={'shape':[rows,cols],'equal':torch.equal(a,b),'different':(a!=b).sum().item(),'elements':a.numel(),
                 'max_abs':(a.float()-b.float()).abs().max().item(),
                 'original_ms':time(lambda:bnb.matmul_4bit(x,weight.t(),quant_state=state)),
                 'fused_ms':time(lambda:fused(x,weight,state,scales))}
            results.append(row)
            print(json.dumps(row),flush=True)
            for splits,group in [(1,16),(0,64),(0,128),(0,256)]:
                c=wmma(x,weight,state,scales,splits,group)
                row={'shape':[rows,cols],'wmma':[splits,group],'equal':torch.equal(a,c),
                     'different':(a!=c).sum().item(),'elements':a.numel(),'max_abs':(a.float()-c.float()).abs().max().item(),
                     'fused_ms':time(lambda:wmma(x,weight,state,scales,splits,group))}
                results.append(row)
                print(json.dumps(row),flush=True)
    (ROOT/'artifacts'/'nf4-kernel-probe.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
