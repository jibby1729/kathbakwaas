"""CPU FP32 thread-count probe, comparing every output logit."""
drafts=[row['draft'] for row in json.loads((ROOT/'artifacts/register-model.json').read_text(encoding='utf-8')) if row['mode']=='baseline']
restorer=engine.restorer
encoded_drafts=[torch.tensor([restorer._encode(engine.strip_marks(draft),restorer.config['default_style'])]) for draft in drafts]
torch.set_num_threads(4)
with torch.inference_mode():
    reference_logits=[restorer.model(ids) for ids in encoded_drafts]
    restorer_results=[]
    for threads in (2,4,6,8,12,16):
        torch.set_num_threads(threads)
        for index,ids in enumerate(encoded_drafts):
            result=restorer.model(ids)
            durations=[]
            for _ in range(3):
                started=time.perf_counter();result=restorer.model(ids);durations.append((time.perf_counter()-started)*1000)
            row={'threads':threads,'draft_index':index,'chars':ids.shape[1],
                 'identical_logits':torch.equal(result,reference_logits[index]),
                 'different_logits':int((result!=reference_logits[index]).sum()),'ms':durations}
            restorer_results.append(row)
        print('RESTORER_THREADS',threads,'long_ms',restorer_results[-len(drafts)]['ms'],
              'all_logits_identical',all(r['identical_logits'] for r in restorer_results[-len(drafts):]),flush=True)
torch.set_num_threads(4)
(ROOT/'artifacts/restorer-threads.json').write_text(json.dumps(restorer_results,indent=2))
print('RESTORER_THREADS_DONE',flush=True)
