"""Exact-length prefill graphs; no padded attention or cached answers."""
from collections import OrderedDict
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

class PrefillGraphs:
    def __init__(self,decoder,max_graphs=2):
        self.decoder=decoder
        self.original=decoder.forward
        self.graphs=OrderedDict()
        self.max_graphs=max_graphs
        self.pool=torch.cuda.graph_pool_handle()
        decoder.forward=self.forward

    def forward(self,*args,**kwargs):
        ids=kwargs.get('input_ids');past=kwargs.get('past_key_values')
        if (ids is None or ids.ndim!=2 or ids.shape[0]!=4 or not 1<ids.shape[1]<self.decoder.cache.capacity
                or (past is not None and past.get_seq_length()!=0)):
            return self.original(*args,**kwargs)
        length=ids.shape[1]
        if length not in self.graphs and torch.cuda.mem_get_info()[0] < 1536 * 2**20:
            return self.original(*args,**kwargs)
        self.decoder.cache.cursor=0
        if length not in self.graphs:
            if len(self.graphs)>=self.max_graphs:
                self.graphs.popitem(last=False)
            inputs=ids.contiguous().clone();positions=kwargs['position_ids'].clone()
            call=dict(input_ids=inputs,position_ids=positions,
                      attention_mask={'full_attention':None,'sliding_attention':None},
                      past_key_values=self.decoder.cache,use_cache=True,logits_to_keep=1,return_dict=True)
            stream=self.decoder.warmup_stream;stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.inference_mode():
                self.decoder.original(**call)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool),torch.inference_mode():
                output=self.decoder.original(**call)
            self.graphs[length]=(graph,inputs,positions,output.logits)
        graph,inputs,positions,logits=self.graphs[length]
        self.graphs.move_to_end(length)
        inputs.copy_(ids);positions.copy_(kwargs['position_ids'])
        graph.replay()
        self.decoder.cache.cursor=length
        return CausalLMOutputWithPast(logits=logits,past_key_values=self.decoder.cache)
