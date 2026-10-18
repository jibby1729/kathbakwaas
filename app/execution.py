"""Reuse exact NF4 conversions and CUDA launches without changing decoding math."""
import logging
import math

logger = logging.getLogger(__name__)


class CachedLinear:
    def __init__(self, module):
        import bitsandbytes as bnb
        self.original = module.forward
        self.dtype = module.compute_dtype
        self.bias = module.bias
        # Exactly the dequantization/transpose used by bnb MatMul4Bit.forward.
        self.weight = bnb.functional.dequantize_4bit(
            module.weight.t(), module.weight.quant_state,
        ).to(self.dtype).t()

    def __call__(self, x):
        import torch
        # Preserve the original prefill path, including its single-vector GEMV.
        if tuple(x.shape[:2]) != (4, 1) or x.ndim != 3 or torch.is_grad_enabled():
            return self.original(x)
        return torch.nn.functional.linear(x.to(self.dtype), self.weight, self.bias).to(x.dtype)


class MLPGraph:
    def __init__(self, module, sample, pool):
        import torch
        self.original = module.forward
        self.input = sample.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                reference = self.original(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool), torch.inference_mode():
            self.output = self.original(self.input)
        self.graph.replay()
        if not torch.equal(reference, self.output):
            raise RuntimeError('CUDA graph did not reproduce the original MLP output')

    def __call__(self, x):
        import torch
        if (x.shape != self.input.shape or x.dtype != self.input.dtype
                or x.device != self.input.device or torch.is_grad_enabled()):
            return self.original(x)
        self.input.copy_(x)
        self.graph.replay()
        return self.output


def optimize_execution(model, cache_mb=1000):
    """Called once during startup; all request execution remains serialized.

    Leave 2 GiB before graph capture for KV caches and longer requests.
    A failed optional setup is rolled back to the original implementation.
    """
    import torch
    import bitsandbytes as bnb
    patches = []
    with torch.inference_mode():
        try:
            free, _ = torch.cuda.mem_get_info()
            budget = min(cache_mb * 2**20, max(0, free - 2048 * 2**20))
            used = 0
            linears = sorted(
                [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)],
                key=lambda m: math.prod(m.weight.quant_state.shape),
            )
            for module in linears:
                size = math.prod(module.weight.quant_state.shape) * torch.tensor([], dtype=module.compute_dtype).element_size()
                if used + size > budget:
                    break
                cached = CachedLinear(module)
                patches.append((module, module.forward))
                module.forward = cached
                used += size
            count = len(patches)
            pool = torch.cuda.graph_pool_handle()
            sample = torch.ones((4, 1, model.config.hidden_size), device=model.device, dtype=model.dtype)
            for layer in model.model.layers:
                graph = MLPGraph(layer.mlp, sample, pool)
                patches.append((layer.mlp, layer.mlp.forward))
                layer.mlp.forward = graph
            logger.info('Execution reuse ready: %d exact NF4 conversions (%.0f MiB), %d MLP CUDA graphs',
                        count, used / 2**20, len(model.model.layers))
            return {'cached_linears': count, 'cache_mb': round(used / 2**20), 'mlp_graphs': len(model.model.layers)}
        except Exception:
            for module, original in reversed(patches):
                module.forward = original
            logger.exception('Optional execution reuse unavailable; retaining original inference')
            return {'cached_linears': 0, 'cache_mb': 0, 'mlp_graphs': 0}
