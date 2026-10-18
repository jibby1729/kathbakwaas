"""CUDA graph microbenchmarks, excluding Python launch overhead, with exactness gates."""
import ctypes,json,itertools,gc
from pathlib import Path
import torch
import bitsandbytes as bnb
ROOT=Path(__file__).resolve().parent.parent
lib=ctypes.CDLL(str(ROOT/'artifacts/nf4_palette.dll'))
lib.make_palette.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p]
lib.launch_palette.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*2+[ctypes.c_void_p]+[ctypes.c_int]*3

def palette_for(weight,state):
    scales=bnb.functional.dequantize_blockwise(state.absmax,state.state2)+state.offset
    palette=torch.empty((scales.numel(),16),device=weight.device,dtype=torch.bfloat16)
    rc=lib.make_palette(scales.data_ptr(),state.code.data_ptr(),palette.data_ptr(),scales.numel(),torch.cuda.current_stream().cuda_stream)
    if rc:raise RuntimeError(rc)
    return scales,palette

def product(x,w,state,scales,palette,warps,tile,use_palette):
    rows,cols=state.shape
    out=torch.empty((4,1,rows),device=x.device,dtype=x.dtype)
    rc=lib.launch_palette(x.data_ptr(),w.data_ptr(),palette.data_ptr(),scales.data_ptr(),state.code.data_ptr(),out.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream,warps,tile,use_palette)
    if rc:raise RuntimeError(rc)
    return out

def graph_time(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):out=fn()
    for _ in range(3):graph.replay()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)/100

if __name__=='__main__':
    records=[]
    torch.manual_seed(1729)
    with torch.inference_mode():
        for rows,cols in [(1024,2560),(2048,2560),(2560,2048),(10240,2560),(2560,10240)]:
            dense=torch.randn((rows,cols),device='cuda',dtype=torch.bfloat16)*.02
            w,state=bnb.functional.quantize_4bit(dense,compress_statistics=True,quant_type='nf4')
            del dense
            scales,palette=palette_for(w,state)
            expanded=bnb.functional.dequantize_4bit(w,state)
            x=torch.randn((4,1,cols),device='cuda',dtype=torch.bfloat16)
            ref=bnb.matmul_4bit(x,w.t(),quant_state=state)
            for label,fn in [('baseline',lambda:bnb.matmul_4bit(x,w.t(),quant_state=state)),('expanded',lambda:torch.nn.functional.linear(x,expanded))]:
                row={'shape':[rows,cols],'mode':label,'ms':graph_time(fn),'equal':torch.equal(ref,fn())}
                records.append(row);print(json.dumps(row),flush=True)
            for use_palette,warps,tile in itertools.product([0,1],[1,2,4],[32,64,128,256]):
                fn=lambda:product(x,w,state,scales,palette,warps,tile,use_palette)
                out=fn()
                row={'shape':[rows,cols],'mode':'kernel','palette':bool(use_palette),'warps':warps,'tile':tile,
                     'equal':torch.equal(ref,out),'different':int((ref!=out).sum()),'ms':graph_time(fn)}
                records.append(row)
            best=sorted([r for r in records if r['shape']==[rows,cols]],key=lambda r:r['ms'])[:6]
            print('BEST '+json.dumps(best),flush=True)
            (ROOT/'artifacts/palette-probe.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            gc.collect();torch.cuda.empty_cache()
