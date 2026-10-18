"""Exact beam gathers captured together, with identity-only warm-up."""
import torch

class ReorderGraphs:
    def __init__(self,cache):
        self.cache=cache
        self.original=cache.reorder_cache
        self.indices=torch.arange(4,device=cache.layers[0].keys.device,dtype=torch.int32)
        self.identity=self.indices.clone()
        self.graphs={}
        self.pool=torch.cuda.graph_pool_handle()
        cache.reorder_cache=self.reorder

    def gather(self,cursor):
        for layer in self.cache.layers:
            for buffer in (layer.keys,layer.values):
                used=buffer[:,:,:cursor]
                used.copy_(used.index_select(0,self.indices))

    def reorder(self,indices):
        cursor=self.cache.cursor
        if cursor not in self.graphs:
            # Warm/capture identity gathers: do not apply the requested reorder
            # until replay, since repeating a real reorder changes beam history.
            self.indices.copy_(self.identity)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.inference_mode():
                self.gather(cursor)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool),torch.inference_mode():
                self.gather(cursor)
            self.graphs[cursor]=graph
        self.indices.copy_(indices)
        self.graphs[cursor].replay()
