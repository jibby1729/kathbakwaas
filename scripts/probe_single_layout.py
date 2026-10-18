"""Keep one lossless NF4 layout, reconstructing original bytes for prefill."""
unpacklib=ctypes.CDLL(str(ROOT/'artifacts/nf4_unpack.dll'))
unpacklib.unpack_nf4.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_void_p]
def unpack(module):
    rows,cols=module.weight.quant_state.shape
    out=torch.empty(module.weight.shape,device=module.weight.device,dtype=torch.uint8)
    rc=unpacklib.unpack_nf4(module._register_weights.data_ptr(),out.data_ptr(),rows,cols,torch.cuda.current_stream().cuda_stream)
    if rc:raise RuntimeError(rc)
    return out
layout_checks=[]
with torch.inference_mode():
    for name,m in engine.model.named_modules():
        if not isinstance(m,bnb.nn.Linear4bit):continue
        reconstructed=unpack(m)
        equal=torch.equal(m.weight,reconstructed)
        layout_checks.append({'name':name,'equal':equal})
        assert equal,name
        m.weight.data=m._register_weights.view(torch.uint8).view(m.weight.shape)
        m._single_layout_decode=m.forward
        def single_layout_forward(self,x):
            if x.shape==(4,1,self.in_features):return self._single_layout_decode(x)
            original_layout=unpack(self)
            packed_layout=self.weight.data
            try:
                self.weight.data=original_layout
                return self._register_original(x)
            finally:
                self.weight.data=packed_layout
        m.forward=types.MethodType(single_layout_forward,m)
del reconstructed
gc.collect();torch.cuda.empty_cache()
(ROOT/'artifacts/single-layout-checks.json').write_text(json.dumps(layout_checks,indent=2))
print('SINGLE_LAYOUT_CHECKS',len(layout_checks),flush=True)
run('single_layout')
run('single_layout_repeat')
run('single_layout_prefilled',session='typing-probe')
print('SINGLE_LAYOUT_READY',flush=True)
