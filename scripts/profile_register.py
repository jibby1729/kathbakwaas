"""Run inside the warmed interactive probe."""
import functools
measurements={}
def measure(obj,name,label):
    original=getattr(obj,name)
    @functools.wraps(original)
    def wrapped(*args,**kwargs):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();start=time.perf_counter()
        result=original(*args,**kwargs)
        elapsed=(time.perf_counter()-start)*1000;b.record()
        measurements.setdefault(label,[]).append((a,b,elapsed))
        return result
    setattr(obj,name,wrapped)
    return original
old_forward=measure(engine.model,'forward','forward')
old_reorder=measure(decoder.cache,'reorder_cache','reorder')
run('profile_events')
torch.cuda.synchronize()
print({key:{'count':len(items),'gpu_ms':sum(a.elapsed_time(b) for a,b,c in items),'cpu_ms':sum(c for a,b,c in items)} for key,items in measurements.items()})
engine.model.forward=old_forward;decoder.cache.reorder_cache=old_reorder
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    run('profile_trace')
prof.export_chrome_trace(str(ROOT/'artifacts/register-trace.json'))
report=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35)
(ROOT/'artifacts/register-profile.txt').write_text(report,encoding='utf-8')
print(report)
