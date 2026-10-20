"""Reproduce retired-pool failure, or stress the fixed capture/release lifecycle."""
import argparse
import gc
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from app.graph_pool import capture_pool

parser = argparse.ArgumentParser()
parser.add_argument('--legacy', action='store_true')
args = parser.parse_args()
owner = SimpleNamespace(graphs={}, pool=torch.cuda.graph_pool_handle())
inputs = torch.arange(4096, device='cuda', dtype=torch.float32)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    expected = inputs * 2 + 1
torch.cuda.current_stream().wait_stream(stream)
held_output = None
try:
    for index in range(100):
        graph = torch.cuda.CUDAGraph()
        pool = owner.pool if args.legacy else capture_pool(owner)
        with torch.cuda.graph(graph, pool=pool):
            output = inputs * 2 + 1
        owner.graphs[index] = graph
        graph.replay()
        assert torch.equal(output, expected)
        # Retain an output tensor, like prompt-cache snapshots or caller results,
        # while releasing the graph. empty_cache cannot revive this pool.
        held_output = output
        torch.cuda.synchronize()
        owner.graphs.clear()
        del graph
        gc.collect()
        torch.cuda.empty_cache()
    if args.legacy:
        raise AssertionError('Legacy lifecycle unexpectedly passed; reproduction inconclusive')
    print('PASS: 100 capture/replay/release cycles, exact outputs throughout', flush=True)
except RuntimeError as exc:
    if args.legacy and 'use_count > 0' in str(exc):
        print('REPRODUCED legacy retired-pool failure:', str(exc), flush=True)
    else:
        raise
