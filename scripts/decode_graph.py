"""Experimental shape-exact decoder graphs; deliberately not enabled in the app."""
import functools
import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer
from transformers.modeling_outputs import CausalLMOutputWithPast


class BufferLayer(DynamicLayer):
    def __init__(self, owner, shape, dtype, device):
        super().__init__()
        self.owner = owner
        self.keys = torch.zeros(shape,dtype=dtype,device=device)
        self.values = torch.zeros_like(self.keys)
        self.is_initialized = True
        self.dtype,self.device = dtype,device

    def get_seq_length(self):
        return self.owner.cursor

    def update(self,k,v,*args,**kwargs):
        start = self.owner.cursor
        end = start+k.shape[-2]
        self.keys[:,:,start:end].copy_(k)
        self.values[:,:,start:end].copy_(v)
        # Match the contiguous tensors of the original DynamicCache, including
        # attention's exact logical dimensions (no extra masked padding).
        return self.keys[:,:,:end].contiguous(),self.values[:,:,:end].contiguous()

    def reorder_cache(self,beam_idx):
        self.keys.copy_(self.keys.index_select(0,beam_idx))
        self.values.copy_(self.values.index_select(0,beam_idx))


class BufferCache(DynamicCache):
    def __init__(self,model,capacity=128):
        super().__init__()
        self.cursor = 0
        self.capacity = capacity
        config = model.config
        shape = (4,config.num_key_value_heads,capacity,config.head_dim)
        self.layers = [BufferLayer(self,shape,model.dtype,model.device) for _ in range(config.num_hidden_layers)]

    def import_from(self,cache):
        self.cursor = int(cache.get_seq_length())
        for dest,src in zip(self.layers,cache.layers):
            dest.keys[:,:,:self.cursor].copy_(src.keys)
            dest.values[:,:,:self.cursor].copy_(src.values)

    def export_to_dynamic(self,model):
        cache = DynamicCache(config=model.config)
        for dest,src in zip(cache.layers,self.layers):
            dest.update(src.keys[:,:,:self.cursor].contiguous(),src.values[:,:,:self.cursor].contiguous())
        return cache


class DecoderGraphs:
    def __init__(self,model,capacity=128):
        self.original = model.forward
        self.cache = BufferCache(model,capacity)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}
        self.ids = torch.zeros((4,1),device=model.device,dtype=torch.long)
        self.positions = torch.zeros_like(self.ids)
        self.model = model
        @functools.wraps(self.original)
        def wrapped(*args,**kwargs):
            return self.forward(*args,**kwargs)
        model.forward = wrapped

    def forward(self,*args,**kwargs):
        ids = kwargs.get('input_ids')
        past = kwargs.get('past_key_values')
        if (ids is not None and ids.shape == (4,1) and past is not None
                and past is not self.cache and 0 < past.get_seq_length() < self.cache.capacity):
            self.cache.import_from(past)
            past = self.cache
        if ids is None or ids.shape != (4,1) or past is not self.cache:
            result = self.original(*args,**kwargs)
            if (result.past_key_values is not None and ids is not None and ids.shape[0] == 4
                    and 0 < result.past_key_values.get_seq_length() < self.cache.capacity):
                self.cache.import_from(result.past_key_values)
                result.past_key_values = self.cache
            return result
        cursor = self.cache.cursor
        if cursor >= self.cache.capacity:
            # Preserve all accumulated context and continue through the original
            # dynamic/sliding cache. Capacity limits acceleration, not output.
            fallback = dict(kwargs,past_key_values=self.cache.export_to_dynamic(self.model))
            return self.original(*args,**fallback)
        self.ids.copy_(ids)
        self.positions.copy_(kwargs['position_ids'])
        if cursor not in self.graphs:
            call = dict(input_ids=self.ids,position_ids=self.positions,
                        attention_mask={'full_attention':None,'sliding_attention':None},
                        past_key_values=self.cache,use_cache=True,logits_to_keep=1,return_dict=True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.inference_mode():
                self.original(**call)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool),torch.inference_mode():
                output = self.original(**call)
            self.graphs[cursor] = graph,output.logits
        graph,logits = self.graphs[cursor]
        graph.replay()
        self.cache.cursor += 1
        return CausalLMOutputWithPast(logits=logits,past_key_values=self.cache)
