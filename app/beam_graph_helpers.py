"""Capture the original HF beam helpers, retaining their operations and scores."""
import functools
from collections import OrderedDict
import torch

class BeamHelperGraph:
    def __init__(self,model,name,normalize=False,max_graphs=4,stream=None):
        self.original=getattr(model,name)
        self.normalize=normalize
        self.graphs=OrderedDict()
        self.max_graphs=max_graphs
        self.pool=torch.cuda.graph_pool_handle()
        self.warmup_stream=stream if stream is not None else torch.cuda.Stream()
        setattr(model,name,self.call)

    def call(self,**kwargs):
        reference_kwargs=kwargs.copy()
        if self.normalize:
            denominator=(kwargs['cur_len']+1-kwargs['decoder_prompt_len'])**kwargs['length_penalty']
            kwargs['topk_log_probs']=kwargs['topk_log_probs']/denominator
            # Original helper still performs its division, now exactly by one.
            kwargs.update(cur_len=0,decoder_prompt_len=0,length_penalty=0.0)
        names=[name for name,value in kwargs.items() if isinstance(value,torch.Tensor)]
        key=tuple((name,tuple(value.shape),value.dtype) if isinstance(value,torch.Tensor) else (name,value)
                  for name,value in kwargs.items())
        if key not in self.graphs:
            if torch.cuda.mem_get_info()[0] < 1536 * 2**20:
                return self.original(**reference_kwargs)
            if len(self.graphs)>=self.max_graphs:
                self.graphs.popitem(last=False)
            inputs={name:kwargs[name].clone() for name in names}
            call=kwargs.copy();call.update(inputs)
            expected=self.original(**reference_kwargs)
            stream=self.warmup_stream;stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.inference_mode():
                self.original(**call)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool),torch.inference_mode():
                outputs=self.original(**call)
            graph.replay()
            if not all(torch.equal(a,b) for a,b in zip(expected,outputs)):
                raise RuntimeError('Beam helper graph failed exact numerical validation')
            self.graphs[key]=(graph,inputs,outputs)
        graph,inputs,outputs=self.graphs[key]
        self.graphs.move_to_end(key)
        torch._foreach_copy_([inputs[name] for name in names],[kwargs[name] for name in names])
        graph.replay()
        return outputs
