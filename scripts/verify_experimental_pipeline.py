"""Full token/text comparisons, including continuation beyond graph capacity."""
importlib.reload(decode_graph)
decoder.cache.export_to_dynamic=types.MethodType(decode_graph.BufferCache.export_to_dynamic,decoder.cache)
prefills.original=types.MethodType(decode_graph.DecoderGraphs.forward,decoder)
verification_texts=[source,'How are you?','Please close the window.',source+' right now',
    "I will arrive tomorrow at 5 o'clock. Please wait for me near the bridge.",
    'My grandmother likes drinking tea in the morning.','Please bring two cups of tea.',
    'I cannot come today, but I can come tomorrow.','Where is the nearest hospital?',
    'My name is Aamir. I live in Srinagar.','Is this yours or mine?',
    'Please do not forget to bring my notebook.',
    'Please tell my brother that I will arrive tomorrow morning. Ask him to wait for me near the bridge, '
    'and bring some tea and bread for our grandmother. If it rains, we can sit inside the house and talk until the rain stops.']
saved_modules=[(m,m.forward) for m in engine.model.modules() if isinstance(m,bnb.nn.Linear4bit)]
saved_forward=engine.model.forward
saved_running=engine.model._get_running_beams_for_next_iteration
saved_finished=engine.model._update_finished_beams
verification=[]
def save_verification():
    (ROOT/'artifacts/full-pipeline-checks.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
def original_linear(self,x):
    original_layout=unpack(self);packed_layout=self.weight.data
    try:
        self.weight.data=original_layout
        return self._register_original(x)
    finally:self.weight.data=packed_layout
try:
    engine.model.forward=decoder.original
    engine.model._get_running_beams_for_next_iteration=beam_running.original
    engine.model._update_finished_beams=beam_finished.original
    for m,_ in saved_modules:m.forward=types.MethodType(original_linear,m)
    for text in verification_texts:
        row=run('verification_original',text)
        verification.append({'phase':'baseline','source':text,'tokens':tokens,**row})
        assert row['identical_tokens'] and row['identical_text']
        save_verification()
finally:
    for m,forward in saved_modules:m.forward=forward
    engine.model.forward=saved_forward
    engine.model._get_running_beams_for_next_iteration=saved_running
    engine.model._update_finished_beams=saved_finished
try:
    assert kernel32.SetProcessAffinityMask(process,masks['performance_physical'])
    for text in verification_texts:
        run('verification_capture',text)
        row=run('verification_optimized',text)
        verification.append({'phase':'optimized','source':text,'tokens':tokens,**row})
        save_verification()
        assert row['identical_tokens'] and row['identical_text'],text
finally:
    assert kernel32.SetProcessAffinityMask(process,old_mask.value)
print('FULL_PIPELINE_VERIFIED',len(verification_texts),flush=True)
