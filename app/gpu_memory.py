"""Drop optional accelerators before admitting work with little GPU headroom."""
import gc
import logging

logger = logging.getLogger(__name__)
RESERVE_MIB = 1536


class GPUMemoryBusy(RuntimeError):
    pass


def release_graphs(backend, include_decoder=True):
    if not backend:
        return
    names = ('decoder', 'reorder', 'prefill') if include_decoder else ('prefill',)
    for name in names:
        backend[name].graphs.clear()
    for helper in backend['beam_helpers']:
        helper.graphs.clear()


def ensure_headroom(engine):
    """Called under the app's single inference lock, outside graph execution.

    This is admission control, not an OOM guarantee: other GPU applications can
    allocate memory after this check. Never shrink beams, input or output limits.
    """
    import torch
    free, _ = torch.cuda.mem_get_info()
    if free >= RESERVE_MIB * 2**20:
        return
    torch.cuda.synchronize()
    backend = getattr(engine, 'fast_execution', None)
    # Prompt shapes and saved inputs are disposable. Retain the bounded decode
    # accelerator when clearing them alone restores enough headroom.
    release_graphs(backend, include_decoder=False)
    engine.prompt_cache.entries.clear()
    gc.collect()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < RESERVE_MIB * 2**20:
        release_graphs(backend)
        gc.collect()
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
    logger.warning('Released optional caches for GPU headroom; %.0f MiB free', free / 2**20)
    if free < RESERVE_MIB * 2**20:
        raise GPUMemoryBusy('GPU memory is busy. Close another GPU application and try again.')
