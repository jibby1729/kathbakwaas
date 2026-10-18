"""Group independent projections without changing each dot product."""
import torch
from types import SimpleNamespace

class GroupedProjections:
    def __init__(self,modules,product,warps=1):
        self.modules=modules;self.product=product;self.warps=warps
        self.originals=[module.forward for module in modules]
        self.sizes=[module.out_features for module in modules]
        self.in_features=modules[0].in_features
        self.weights=torch.cat([module._register_weights for module in modules])
        self.scales=torch.cat([module._register_scales for module in modules])
        self.state=SimpleNamespace(shape=(sum(self.sizes),self.in_features),code=modules[0].weight.quant_state.code)
        self.outputs=None;self.input_pointer=None
        # Preserve existing single-layout fallback views and release duplicates.
        offset=scale_offset=0
        for module in modules:
            size=module._register_weights.numel();scale_size=module._register_scales.numel()
            module._register_weights=self.weights[offset:offset+size]
            module._register_scales=self.scales[scale_offset:scale_offset+scale_size]
            module.weight.data=module._register_weights.view(torch.uint8).view(module.weight.shape)
            offset+=size;scale_offset+=scale_size

    def combined(self,x):
        result=self.product(x,self.weights,self.scales,self.state,self.warps,16)
        return tuple(part.contiguous() for part in result.split(self.sizes,dim=-1))

    def install(self):
        for index,module in enumerate(self.modules):
            def forward(x,index=index):
                if x.shape!=(4,1,self.in_features):return self.originals[index](x)
                if index==0:
                    self.outputs=self.combined(x);self.input_pointer=x.data_ptr()
                if self.input_pointer!=x.data_ptr():return self.originals[index](x)
                return self.outputs[index]
            module.forward=forward
