"""A pool identifier cannot be reused after its last graph is released."""
import torch


def capture_pool(owner):
    # Emptying a graph cache retires its allocator pool even if captured output
    # tensors are still referenced elsewhere. The handle alone does not keep it
    # alive. Start a new pool before capturing the first replacement graph.
    if not owner.graphs:
        owner.pool = torch.cuda.graph_pool_handle()
    return owner.pool
