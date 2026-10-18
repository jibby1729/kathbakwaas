"""Check whether graph bucketing preserves the actual CUDA attention output."""
import json
from pathlib import Path
import torch
from torch.nn.functional import scaled_dot_product_attention as attention

ROOT=Path(__file__).resolve().parent.parent
rows=[]
with torch.inference_mode():
    for seed in (1729,2026):
        torch.manual_seed(seed)
        for length in (1,16,17,32,37,63,64,65,96,127,128,129,255):
            bucket=next(size for size in (32,64,128,256) if size>=length)
            for amplitude in (.2,1,3):
                q=torch.randn(4,8,1,256,device='cuda',dtype=torch.bfloat16)*amplitude
                k=torch.randn(4,4,length,256,device='cuda',dtype=torch.bfloat16)*amplitude
                v=torch.randn_like(k)
                expected=attention(q,k,v,enable_gqa=True)
                kp=torch.zeros(4,4,bucket,256,device='cuda',dtype=torch.bfloat16);vp=torch.zeros_like(kp)
                kp[:,:,:length]=k;vp[:,:,:length]=v
                mask=(torch.arange(bucket,device='cuda')<length).view(1,1,1,-1)
                for mode in ('repeated_kv','gqa'):
                    actual=attention(q,kp.repeat_interleave(2,1) if mode=='repeated_kv' else kp,
                                     vp.repeat_interleave(2,1) if mode=='repeated_kv' else vp,
                                     attn_mask=mask,enable_gqa=mode=='gqa')
                    rows.append({'seed':seed,'length':length,'bucket':bucket,'amplitude':amplitude,'mode':mode,
                                 'identical':torch.equal(expected,actual),'different':int((expected!=actual).sum()),
                                 'max_abs':float((expected.float()-actual.float()).abs().max())})
(ROOT/'artifacts/attention-padding.json').write_text(json.dumps(rows,indent=2))
print(json.dumps({'checks':len(rows),'identical':sum(r['identical'] for r in rows),'examples':[r for r in rows if not r['identical']][:6]},indent=2))
