# Inference optimization investigation

Target: fresh output generation under one second, preserving the pinned NF4 model, BF16 computation, four beams, length penalty 0.7, stopping behavior, and CPU FP32 diacritizer. The target is **not achieved**. Cached translation lookups do not count.

## October 17, 2026

Hardware: RTX 4060 Laptop, 8 GiB; Windows; torch 2.11.0+cu128, transformers 5.20.1, bitsandbytes 0.49.2. Tests run with the web service stopped. These are local measurements with ordinary laptop timing variation, not a controlled hardware performance study.

The main test is the user's sentence: “oh my god you are driving me crazy and you are so annoying and i can't stand you”. It generates 37 tokens. Timing separates generation from CPU diacritization and excludes completed-result caching.

| Experiment | Evidence | Decision |
| --- | --- | --- |
| CPU threads 1, 2, 4 | About 5.1–6.1 seconds total on the long sentence; one thread slows the restorer | Keep four threads |
| Cache exact NF4 dequantization, 2270 MiB | Warm run 3.88 seconds total, same output | Useful, bounded cache |
| Cached dequantization + 34 MLP CUDA graphs | Long sentence generation 3.280/3.282 seconds; total 3.434/3.424 seconds; all four test passages match | Integrate with a smaller 2000 MiB cap and verify |
| Whole-decoder graphs, exact attention lengths | Long sentence generation 3.232/3.287 seconds; capture itself takes 9.68 seconds total | Little benefit over simpler MLP graphs; experimental only |
| Fused four-vector NF4 scalar reduction | Faster matrix products, but 5–46 matrix outputs differ in the probes | Rejected: numerical mismatch |
| Fused tensor-core NF4, sequential 16-element accumulation | 714 real-matrix checks match exactly; four full passages match | Numerically promising, but current implementation slower (about 4 seconds generation), experimental only |

The profiler records 32,798 CUDA launches over eight generation iterations. NF4 dequantization takes about 40% and matrix multiplication about 49% of the recorded GPU time. CPU launch/dispatch work is also substantial. These percentages come from a profiled run with instrumentation overhead; do not use its wall time as an unprofiled latency benchmark.

## Earlier conversion-cache backend (retained as fallback)

`app/execution.py` caches the exact BF16 weight tensors that bitsandbytes would produce, and replays the original MLP operations through CUDA graphs. It applies only to four-beam single-token decode shapes. Prefill and single-vector execution retain the original code path. No model weights are requantized; arithmetic precision and beam search stay unchanged.

The cache uses at most 2000 MiB and reserves 1.5 GiB of reported free memory before setup. Graph capture is performed at startup, checked against the original MLP result, and rolled back if setup fails. The existing single inference lock also serializes access to graph buffers. Set `KOSHUR_OPTIMIZE_EXECUTION=0` to use the original path for comparison.

`scripts/verify_execution.py` compares full generated token IDs and both text outputs for ten passages, repeats the long sentence, and exercises prefix caching and cancellation. Its results, rather than this description, are authoritative for validation status.

Verification completed: all ten passages, two repeated runs, the prefix-reuse run, and the run after a pre-cancelled request matched full baseline token IDs and both text outputs. The app setup cached 155 matrices (1970 MiB) and captured 34 MLP graphs. Long-sentence generation was 3.565 seconds on the first optimized pass and 3.519/3.460 seconds on repeats. The prefix run took 3.705 seconds and the final post-cancellation run 4.715 seconds, so substantial timing variability remains. All 17 API/cache tests and the browser typing test passed. This is numerical/output regression coverage, not proof of translation quality across all possible inputs, and not proof of the latency goal.

## Reproduction and retained evidence

Use the project `.venv` and stop the GPU web service before benchmarks. Do not run these together on the same GPU.

- `scripts/profile_decode.py`: original decoder CPU/CUDA trace.
- `scripts/benchmark_execution.py`: thread counts, exact weight cache, MLP graphs.
- `scripts/benchmark_full_graph.py` and `scripts/decode_graph.py`: bounded, shape-specific whole-decoder graph prototype, now with a dynamic-cache fallback beyond its acceleration capacity.
- `scripts/probe_nf4_kernel.py` and `scripts/nf4_four_beam.cu`: isolated fused-kernel numerical/timing probes. Requires the local compiled `artifacts/nf4_four_beam.dll`, CUDA 12.9 and Visual Studio C++ tools. Never imported by the app.
- `scripts/benchmark_fused_graph.py`: every real quantized matrix checked at three activation scales, then end-to-end comparison.
- `scripts/verify_execution.py`: app optimization verification against the baseline.

JSON reports and profiler traces live under ignored `artifacts/`. In particular: `execution-benchmark.json`, `full-graph-benchmark.json`, `nf4-kernel-probe.json`, `fused-real-weight-checks.json`, `fused-graph-benchmark.json`, and `execution-verification.json`.

## Remaining investigation

The sub-second requirement still needs substantially faster matrix products and lower decode-loop overhead. Next candidates are better tiling/pipelining of the numerically matching fused kernel, a lower-memory combination of fused and cached matrices, and measuring beam-search bookkeeping separately. Reject alternatives that change numerical results. Preserve long-input support and cancellation; do not achieve the target by shortening outputs, reducing beams, changing weights/precision, or reusing a completed answer.

## Native NF4 decoder investigation and integration

The native kernel reads a lossless permutation of NF4 bytes directly into BF16 tensor-core operands, using sequential `mma.m16n8k16` accumulation. It does not split the dot-product reduction or requantize weights. Register-shuffle codebook lookup and a coalesced scale layout improve the matrix products. The three attention projections share a launch, but each output retains its original dot product and contiguous tensor layout.

| Experiment | Result |
| --- | --- |
| Direct register NF4 kernel + exact-length decode graphs | 714 real projection checks and six complete passages match exactly; long-sentence generation 1.317–1.361 s |
| Register-shuffle codebook lookup | 714 projection checks pass; long sentence about 1.21–1.23 s generation |
| Captured beam-cache gathers + direct prefix-cache import | Typed-prefix generation 0.994/0.997 s; total 1.15–1.18 s |
| Coalesced scales, grouped attention projections, captured original beam helpers | Isolated warm generation about 0.97–1.01 s; no token/text differences |
| More warps redundantly computing output tiles | Exact results, no useful speed advantage |
| Preexpanded per-block codebooks | Exact results, slower than register lookup; rejected |
| CPU restorer with 6–16 threads | Faster at some counts, but raw logits differ; rejected |
| Four threads on physical performance cores | All logits match on six drafts; long-draft restorer about 0.12–0.14 s in the probe |
| One compressed weight layout | All 238 matrices reconstruct byte-for-byte; removes about 1.5 GiB of live tensor allocation |
| Shared warm-up CUDA stream | Reduces allocator reservation further; fresh app timing still varies and does not reliably meet the goal |

`app/native_decode.py` builds the local CUDA source in `app/kernels/nf4_decode.cu` and validates every projection at three activation scales before installing it. The original NF4 bytes are reconstructed transiently for stock prefill operations. Disk weights, quantization metadata, precision, and generation settings are unchanged. This is an inference-only in-memory layout; the app does not save the transformed model as a checkpoint.

`app/decoder_graph.py` warms 127 exact-position graphs at startup. Its cache capacity limits acceleration, not context or output length: it exports the complete prefix into the original dynamic/sliding cache when needed. `app/reorder_graph.py` captures the original beam gathers with identity-only warm-up. `app/prefill_graph.py` keeps at most eight exact-length prefill graphs, and `app/beam_graph_helpers.py` keeps at most 16 shapes per helper. New prefill shapes still incur capture latency. The original Hugging Face beam-search loop, stopping criteria, four beams, length penalty, and 512-token output limit remain in use.

`app/cpu_execution.py` optionally restricts this Windows process to physical performance cores. It changes neither thread count nor FP32 computation. It does not change system-wide power settings or other processes. Set `KOSHUR_PERFORMANCE_CORES=0` to disable this placement. CPU model kernels are warmed before readiness so their first initialization does not fall on a user's request.

The fresh app backend passed all 13 full token-sequence and text comparisons, including a 107-token translation that continues beyond the graph capacity. Typed-prefix reuse, a cancellation during generation, and the request after cancellation also passed. The single-layout/shared-stream verification took roughly 50 seconds to initialize including graph preparation. Its repeated long-sentence generation was **1.093 and 1.309 s**, typed-prefix generation **1.076 s**, and the post-cancellation request **1.399 s**. Total typed-prefix processing was **1.24 s** including restoration and other work. The 107-token passage still took about **6.54 s total**. These results supersede the faster isolated prototype when describing current app latency. All 17 API/cache tests and the JavaScript typing checks pass.

Evidence: `artifacts/register-model.json`, `register-v2-real-checks.json`, `register-scales-real-checks.json`, `single-layout-checks.json`, `restorer-threads.json`, `cpu-affinity.json`, `full-pipeline-checks.json`, and `native-shared-stream-verification.json`. Full CUDA traces are retained under `artifacts/`. `scripts/create_native_reference.py` produces stock-model reference token sequences; `scripts/verify_native_backend.py` checks the current app backend against them. Stop the web service before either GPU script.

The goal remains open. The isolated prototype crossed one second for generation, but current app timing, first-use prefill capture, larger passages, and restoration still prevent a reliable sub-second experience. Further work should measure driver/graph memory separately from PyTorch tensor allocation, reduce first-use capture cost, and continue exact kernel improvements. Do not count completed-answer cache hits as inference.

The early graph benchmark's `first_step_ms` used a CPU dispatch timestamp and therefore under-reported asynchronous GPU work. The app now records CUDA events for this field. This correction does not change the reported full generation/total times above.

Final live HTTP check after restarting the webpage: the long sentence reused 36 prompt tokens and took **1.379 s HTTP wall time**, **1.189 s generation**, and **0.147 s diacritization**. “How are you?” took **0.669 s** HTTP wall time, and “Please close the window.” **0.633 s**. All three responses had `cached: false` and matched the saved draft and diacritized text. Evidence: `artifacts/native-live-verification.json`. The browser's typing debounce is separate from these request times. The local service was left running on port 8000.

## Broad accuracy and memory audit (supersedes the earlier prefill checks)

The user prioritized unchanged translations and GPU headroom over a sub-second target. The broader audit found a real regression that the earlier small test missed: incremental prompt reuse changed **15 of 161 translations**, including `i am really hungry today and im also annoyed by what you said`. These were draft/token changes, not just different diacritics. The native decoder with a fresh full prompt matched stock on every case.

The fixed corpus in `scripts/accuracy_corpus.py` contains 161 distinct inputs, 7,686 English characters, and 16 paragraphs. It covers informal typing, negation, tense, pronouns, requests, idioms, names, numbers and punctuation. Stock produced 3,748 output tokens. Four complete passes produced 644 translations; all comparisons include the entire returned token sequence, draft, and diacritized text.

| Computation path | Compared with stock | Changed outputs | Errors |
| --- | ---: | ---: | ---: |
| Native decoder, complete fresh prompt | 161 | 0 | 0 |
| Old incremental typing cache, rejected | 161 | 15 | 0 |
| Complete-prompt preparation while typing, replacement | 161 | 0 | 0 |

The replacement uses `generate()` itself to prepare the normal four-beam model inputs and run its complete `_prefill` call, then stops before selecting any generated token. It saves independent logits and KV tensors. Translation can reuse the snapshot only when the complete tokenized input is identical. Adding, deleting, or changing words requires a fresh complete prefill. This preserves the computation's batch and sequence shapes instead of assuming that a mathematically equivalent partial-prefix computation will give the same finite-precision result. The temporary `_prefill` hook is restored in `finally`; the existing GPU lock serializes it. This uses the pinned Transformers version's internal helper and should be revalidated when upgrading dependencies.

The four beams, NF4 weights, BF16 compute, length penalty 0.7, stopping behavior, 512-token output limit, and CPU FP32 diacritizer are unchanged. The complete-prompt cache is capped at four sessions with a two-minute TTL. Copies prevent generation or another prefill from modifying stored snapshots. Prefill now waits for its CUDA stream to finish before reporting readiness/timing.

The default decoder now warms at most **52 positions (24 through 75)**, rather than 127. Positions outside that set run the same operations without a graph. The 128-token buffer still exports full context to the dynamic cache when necessary; it does not truncate output. Prefill graphs are capped at two, and beam-helper shapes at four each. The alternative conversion cache is capped at 1000 MiB, reserving 2 GiB before its graph setup.

Before a new GPU operation, admission control checks for **1536 MiB free**. It first drops saved prompts and disposable prefill/beam-helper graphs; if necessary it also drops decoder/reorder graphs. If reclamation still cannot provide the reserve, it returns a retryable response before inference. Other applications can allocate concurrently, so this is risk reduction, not an absolute OOM guarantee. Graph capture also checks headroom. One model process and one inference lock remain mandatory.

The revised decoder initialized in about 25.5 seconds. Ordinary cases left roughly 2.0-2.1 GiB free in the comparisons. Longer cases exercised reclamation. The rejected 64-graph trial and its 30 partial comparisons are retained separately; the final broad decoder comparison uses 52 graphs. The complete-prompt broad pass used the first, more aggressive cache-reclamation policy; staged reclamation was subsequently covered by the dedicated memory stress run. That change controls cache lifetime, not model arithmetic.

Fresh complete-prompt translation took **1.25 seconds** for the hungry/annoyed sentence and **1.40 seconds** for the driving-me-crazy sentence, including CPU diacritization. With the complete prompt already prepared, their translation stages took **0.88 and 0.98 seconds**, respectively. Preparation work is separate and can overlap typing; these are not completed-answer cache hits. Long paragraphs still take several seconds. Browser debounce and HTTP timing must be distinguished from the model measurements.

These are equality/regression checks on the current local environment, not a proof for every possible input and not a human evaluation of Kashmiri quality. Twelve overlapping references also match the older saved baseline exactly. If the original model chooses a poor word, reproducing it does not make that word correct.

Reproduce with the web service stopped:

```powershell
.venv/Scripts/python.exe scripts/compare_accuracy.py --backend stock
.venv/Scripts/python.exe scripts/compare_accuracy.py --backend native
.venv/Scripts/python.exe scripts/compare_accuracy.py --backend native --typing --report accuracy-native-complete-prompt.json --fail-fast
.venv/Scripts/python.exe scripts/stress_accuracy_memory.py
.venv/Scripts/python.exe scripts/summarize_accuracy.py
```

The rejected incremental path has been removed from the app. Its old source snapshots are retained locally as `artifacts/engine-incremental-before-fix.py` and `artifacts/prefix-cache-before-fix.py`. Reports: `accuracy-stock.json`, `accuracy-native.json`, `accuracy-native-incremental.json`, `accuracy-native-complete-prompt.json`, and readable `accuracy-summary.md`, all under `artifacts/`. The old incremental report cannot be reproduced by simply running the current `--typing` option, which now tests the replacement.

Final stress checks passed for three additional 488-494-character inputs (217-238 output tokens), with exact stock/native token and text equality. Four prompt sessions were populated without exceeding the configured cache capacity. Cancellation during generation and the following request passed. A controlled pressure allocation left about 1099 MiB free; reclamation left 1373 MiB, below the 1536 MiB admission threshold, so the request was safely deferred. After releasing the allocation, the next request again matched stock. CUDA's OOM counter stayed at zero. These are recorded in `artifacts/accuracy-memory-stress.json`; the artificial-pressure sample is separate from normal application memory use. All 21 API/cache/memory tests and the JavaScript typing checks passed.

The restarted webpage passed three fresh HTTP checks (`cached: false`) against stock. The hungry/annoyed sentence took **865 ms of first-use preparation, then 1104 ms for translation over HTTP**; preparation can overlap typing, but these should not be described as 1.1 seconds for the entire cold pipeline. The driving-me-crazy sentence took **1358 ms** for a fresh translation after deliberately preparing a different shorter input; it correctly reused zero prompt tokens. How are you took **383 ms preparation and 452 ms translation**. All three drafts and diacritized outputs matched stock. The final live GPU sample was **6082 MiB used and 1876 MiB free**. Evidence: `artifacts/accuracy-live.json`. The browser reported Ready, and the local service was left running on port 8000 with the corrected cache and a fresh completed-answer cache.

## Expanded equality audit: 400 additional inputs (2026-10-18)

The larger fixed synthetic set has no overlap with the earlier 161 inputs. It contains 160 individually authored sentences, 200 structured sentence combinations and 40 composed paragraphs: 39,662 English characters, 15–293 per input. It covers scope and negation, tense, pronouns, emotions, numbers, place names, informal spelling, punctuation and requests. This is regression coverage, not a human-rated translation benchmark.

Three fresh, sequential model processes generated 1,200 translations. Both optimized paths matched the original stock NF4/four-beam reference on **400/400 full token sequences, drafts and diacritized outputs**, with **zero errors and zero CUDA OOMs**. The typing run performed additions and a deletion before preparing each complete final prompt; **400/400 actually reused that prompt**. No completed-answer cache or curated browser answer participated. No inference code or decoding settings were changed for this audit.

Median translation times were 5.13 s stock, 1.23 s optimized with a fresh prompt, and 0.89 s with an already prepared prompt. Typing preparation is separate work. P95 times were 19.03, 11.64 and 11.62 s respectively, reflecting the longer passages. Minimum after-case sampled free GPU memory was 3869, 1584 and 1468 MiB respectively. Admission control checks headroom before operations; this is not a continuous free-memory floor or a guarantee against another application's allocations.

The readable report is `artifacts/expanded-accuracy-summary.md`; the full inputs and outputs are `expanded-corpus.json`, `expanded-stock.json`, `expanded-native.json` and `expanded-typing.json` under `artifacts/`. Reports record corpus/source hashes, model revisions, library versions and effective generation settings. Reproduction scripts are `scripts/build_expanded_corpus.py`, `scripts/run_expanded_accuracy.py` (stop the web model first) and `scripts/summarize_expanded_accuracy.py`.

The suggested phrase is now “What is your name?”. The previously requested curated Kashmir example remains an intentional browser-only override. General Kashmir-to-Kashir spelling replacement has only been discussed: the requested scope is the place name alone, and no such rule has been implemented.

## 2026-10-18: graph-pool lifetime repair

The live service released optional graphs for GPU headroom, then subsequent
prefill captures failed with PyTorch's `use_count > 0` allocator assertion. Graph
cache clearing retired the shared pool, but the wrappers retained its handle.
That handle alone does not keep a pool alive. Captures now allocate a fresh pool
whenever their graph cache is empty, including after memory-pressure clearing or
eviction of the last entry. Live graphs continue to share their existing pool.
The fix covers prefill, beam helpers, decoder and reorder wrappers without
changing model operations, weights, beams, token limits or diacritization.

`scripts/check_graph_pool.py --legacy` reproduced the exact assertion on this
GPU. The repaired path passed 100 capture/replay/release cycles with exact numeric
outputs, including retained output tensors and allocator-cache cleanup. The 33
Python tests passed. After restarting only the model backend, both “I can speak
Kashmiri” and the version with a full stop translated through the unchanged public
URL in approximately 1.0–1.7 seconds. Twelve deterministic selections from the
saved stock-reference corpus matched both draft and diacritized text exactly;
six used prepared prompts. This additional check compares strings, not token IDs,
and is not a repeat of the full 400-case audit. Evidence is in
`artifacts/graph-repair-public-check.json` and `artifacts/graph-repair-local-check.json`.

The generic HTTP 500 message now identifies a server error instead of suggesting
that shortening a valid sentence will fix it. `/api/health` remains a model-load
readiness signal; it is not an end-to-end inference test. Cloudflare and the
sharing gateway were not restarted during recovery.

## 2026-10-19: competing GPU workload and cleanup of retained outputs

The host process inventory showed the translator using about 4,765 MiB of
dedicated GPU memory and OpenWhispr's separate dictation service using about
1,915 MiB. Free memory was below the unchanged 1,536 MiB admission threshold.
At the user's request, only OpenWhispr's model service was temporarily stopped;
its settings were not edited. Free VRAM rose to about 2,314 MiB and fresh public
translation resumed before the code deployment.

`release_graphs` now also drops the native projection products' last output
tensors and input-pointer markers. Those outputs are only reused within a fused
projection group, but can retain allocator segments after their CUDA graph is
removed. Clearing them between requests releases the references; the first
projection of each group repopulates them before reuse. Weights, numerical
operations, four-beam decoding, token limits and the safety threshold are unchanged.

The small CUDA regression in `scripts/check_graph_release.py` reproduced a
2 MiB retained segment with the old cleanup and zero retained bytes after the
fixed cleanup. This demonstrates the lifecycle defect; it does not establish
that this defect caused all of the live VRAM pressure. All 40 Python tests passed.
After a backend-only restart, 16 uncached API translations (eight with prepared
prompts) matched saved stock draft and diacritized strings exactly. This is a
targeted string comparison, not a new 400-case token audit. Results are in the
ignored `artifacts/memory-recovery-verification.json`.

The original Cloudflare process and gateway remained running. A fresh public
translation of "I can speak Kashmiri." took about 1.22 seconds of server processing;
the CPU female voice returned a valid WAV. Final GPU headroom was 1,926 MiB.
Starting another GPU model can consume this margin again. Host process and port
checks must run outside the restricted sandbox: its isolated process view had
incorrectly made the running service appear absent in earlier diagnostics.
