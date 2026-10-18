"""Temporary process-only core affinity; restore original mask even on error."""
import struct
kernel32=ctypes.WinDLL('kernel32',use_last_error=True)
kernel32.GetCurrentProcess.restype=ctypes.c_void_p
kernel32.GetSystemCpuSetInformation.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong),ctypes.c_void_p,ctypes.c_ulong]
kernel32.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
kernel32.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
process=kernel32.GetCurrentProcess();needed=ctypes.c_ulong()
kernel32.GetSystemCpuSetInformation(None,0,ctypes.byref(needed),process,0)
buffer=ctypes.create_string_buffer(needed.value)
assert kernel32.GetSystemCpuSetInformation(buffer,needed,ctypes.byref(needed),process,0)
cpu_rows=[];offset=0
while offset<needed.value:
    size,kind=struct.unpack_from('II',buffer,offset)
    assert size>=32
    if kind==0:
        ident,group,logical,core,cache,node,efficiency,flags=struct.unpack_from('IHBBBBBB',buffer,offset+8)
        cpu_rows.append(dict(id=ident,group=group,logical=logical,core=core,efficiency=efficiency))
    offset+=size
print('CPU_SETS',cpu_rows,flush=True)
old_mask,system_mask=ctypes.c_size_t(),ctypes.c_size_t()
assert kernel32.GetProcessAffinityMask(process,ctypes.byref(old_mask),ctypes.byref(system_mask))
best_eff=max(r['efficiency'] for r in cpu_rows)
pcores=[r for r in cpu_rows if r['efficiency']==best_eff and r['group']==0]
physical={r['core']:r for r in reversed(pcores)}
masks={'original':old_mask.value,'performance':sum(1<<r['logical'] for r in pcores),
       'performance_physical':sum(1<<r['logical'] for r in physical.values())}
affinity_results=[]
try:
    torch.set_num_threads(4)
    with torch.inference_mode():
        for label,mask in masks.items():
            assert kernel32.SetProcessAffinityMask(process,mask)
            for index,ids in enumerate(encoded_drafts):
                result=restorer.model(ids);durations=[]
                for _ in range(4):
                    started=time.perf_counter();result=restorer.model(ids);durations.append((time.perf_counter()-started)*1000)
                affinity_results.append({'mode':label,'draft_index':index,'identical_logits':torch.equal(result,reference_logits[index]),'ms':durations})
            print('CPU_AFFINITY',label,'long_ms',affinity_results[-len(drafts)]['ms'],
                  'all_identical',all(r['identical_logits'] for r in affinity_results[-len(drafts):]),flush=True)
finally:
    assert kernel32.SetProcessAffinityMask(process,old_mask.value)
(ROOT/'artifacts/cpu-affinity.json').write_text(json.dumps({'cpus':cpu_rows,'results':affinity_results},indent=2))
print('AFFINITY_RESTORED',flush=True)
