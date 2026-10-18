"""Install and warm exact native decoding, with transactional setup rollback."""
import logging
import os

logger=logging.getLogger(__name__)


def optimize_fast_execution(model):
    import torch
    if os.getenv('KOSHUR_NATIVE_DECODE','1')=='0':return None
    if (model.dtype!=torch.bfloat16 or model.config.hidden_size!=2560
            or model.config.num_hidden_layers!=34 or model.config.sliding_window<=128):
        return None
    patches=[]
    try:
        from .native_decode import install_products
        from .decoder_graph import DecoderGraphs
        from .reorder_graph import ReorderGraphs
        from .prefill_graph import PrefillGraphs
        from .beam_graph_helpers import BeamHelperGraph
        with torch.inference_mode():
            torch.cuda.empty_cache()
            free,_=torch.cuda.mem_get_info()
            if free<2300*2**20:
                raise RuntimeError('Insufficient free GPU memory for native decode setup')
            products=install_products(model,patches)
            patches.append((model,'forward',model.forward))
            decoder=DecoderGraphs(model,capacity=128)
            reorder=ReorderGraphs(decoder.cache,stream=decoder.warmup_stream)
            logger.info('Warming up to 52 exact decoder graphs with GPU memory headroom')
            decoder.warmup(reorder,model.config.bos_token_id or 2)
            prefill=PrefillGraphs(decoder)
            beam_helpers=[]
            for name,normalize in [('_get_running_beams_for_next_iteration',False),('_update_finished_beams',True)]:
                patches.append((model,name,getattr(model,name)))
                beam_helpers.append(BeamHelperGraph(model,name,normalize=normalize,stream=decoder.warmup_stream))
            torch.cuda.empty_cache()
        logger.info('Native decoding ready: 238 validated NF4 projections, %d decoder graphs',len(decoder.graphs))
        return {'backend':'native_nf4','products':products,'decoder':decoder,'reorder':reorder,
                'prefill':prefill,'beam_helpers':beam_helpers,'graph_capacity':128}
    except Exception:
        for patch in reversed(patches):
            if callable(patch):
                patch()
            else:
                target,name,original=patch
                setattr(target,name,original)
        logger.exception('Native decoding setup unavailable; retaining the original model forwards')
        return None
