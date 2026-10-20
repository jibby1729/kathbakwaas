"""Verify that projection outputs cannot pin retired CUDA graph pools."""
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from app.gpu_memory import release_graphs


def collect():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_reserved()


def captured_backend():
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # A tiny surviving output can pin an allocator segment after graph release.
        temporary = torch.ones(8 * 1024 * 1024, device='cuda')
        output = temporary[:1].clone()
    graph.replay()
    assert output.item() == 1
    backend = {name: SimpleNamespace(graphs={}) for name in ('decoder', 'reorder', 'prefill')}
    backend['decoder'].graphs[1] = graph
    backend['beam_helpers'] = []
    backend['products'] = [SimpleNamespace(outputs=(output,), input_pointer=123)]
    return backend


torch.cuda.init()
baseline = collect()
backend = captured_backend()
# Reproduce the old cleanup: graph is gone but a projection owns its output.
backend['decoder'].graphs.clear()
retained = collect()
assert retained - baseline >= 2 * 2**20, (baseline, retained)
release_graphs(backend)
released = collect()
assert released < retained and released <= baseline + 2 * 2**20, (baseline, retained, released)
print(f'PASS: old cleanup retained {(retained-baseline)/2**20:.0f} MiB; '
      f'fixed cleanup retains {(released-baseline)/2**20:.0f} MiB', flush=True)
