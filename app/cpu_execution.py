"""Optional process-only placement on physical performance cores on Windows."""
import ctypes
import os
import struct


def prefer_performance_cores():
    if os.name!='nt' or os.getenv('KOSHUR_PERFORMANCE_CORES','1')=='0':
        return None
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.GetSystemCpuSetInformation.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong),ctypes.c_void_p,ctypes.c_ulong]
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    process=kernel.GetCurrentProcess();needed=ctypes.c_ulong()
    kernel.GetSystemCpuSetInformation(None,0,ctypes.byref(needed),process,0)
    if not needed.value:return None
    buffer=ctypes.create_string_buffer(needed.value)
    if not kernel.GetSystemCpuSetInformation(buffer,needed,ctypes.byref(needed),process,0):return None
    rows=[];offset=0
    while offset<needed.value:
        size,kind=struct.unpack_from('II',buffer,offset)
        if size<32 or offset+size>needed.value:return None
        if kind==0:
            ident,group,logical,core,cache,node,efficiency,flags=struct.unpack_from('IHBBBBBB',buffer,offset+8)
            rows.append((group,logical,core,efficiency))
        offset+=size
    if not rows or any(row[0]!=0 for row in rows) or len({row[3] for row in rows})<2:return None
    best=max(row[3] for row in rows)
    physical={row[2]:row for row in reversed(rows) if row[3]==best}
    if len(physical)<4:return None
    original,system=ctypes.c_size_t(),ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(process,ctypes.byref(original),ctypes.byref(system)):return None
    mask=sum(1<<row[1] for row in physical.values()) & original.value
    if mask.bit_count()<4 or not kernel.SetProcessAffinityMask(process,mask):return None
    return {'original_mask':original.value,'mask':mask,'logical_processors':[row[1] for row in physical.values()]}
