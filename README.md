# Kath Bakwaas

English → Kashmiri translation as you type, with optional diacritics and male/female audio playback. Made by [Jibran Iqbal Shah](https://jibby1729.github.io/), inspired by the [KATHE AI 2026 contest](https://www.kaggle.com/competitions/kathe-2026).

The page uses large, readable text, a bigger-text option, copy/paste controls, and three example phrases. Kashmiri appears in Perso-Arabic script. The browser interface is plain HTML, CSS and JavaScript; FastAPI runs the models locally. No external inference API or browser API key is needed.

| Component | Model | Runs on |
| --- | --- | --- |
| Translation | [Koshur-Sarvam-4B](https://huggingface.co/Jibby1729/Koshur-Sarvam-4B) · NF4, four beams | NVIDIA GPU |
| Diacritics | [Koshur-JibDiacritizer-152M](https://huggingface.co/Jibby1729/Koshur-JibDiacritizer-152M) | CPU |
| Speech | [Bolbosh](https://github.com/gaash-lab/Bolbosh) + HiFi-GAN, by [Gaash Lab](https://gaash.nitsri.ac.in/) at NIT Srinagar | Separate CPU-only process |

Translations may contain errors. Published translation benchmark comparisons are on the [model leaderboard](https://huggingface.co/Jibby1729/Koshur-Sarvam-4B#6-comparison-with-other-models); the local regression tests below measure whether optimizations change outputs, not linguistic accuracy.

## Run locally (Windows + NVIDIA GPU)

The tested environment is **Windows, Python 3.13, an RTX 4060 Laptop GPU (8 GB), and CUDA 12.8 PyTorch**. Allow at least 15 GB of free disk space for the translator, with additional room for the separate speech environment. The translator runs in **4-bit NF4** on the GPU; the 152M diacritizer runs on CPU. First startup downloads about 8.4 GB of weights plus tokenizer files. Later starts reuse the cache. Native acceleration additionally needs the CUDA Toolkit and Visual Studio C++ tools; a fallback is available when they are absent.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\start.ps1
```

Open **http://127.0.0.1:8000**. After installation, double-click `start.cmd` to launch again. Keep that terminal open while using the page. Stop with Ctrl+C. `start.ps1 -Port 8001` selects another port.

Wait for **Ready**, then type English. Input processing starts at word boundaries or after a 120 ms gap; translation starts after a 250 ms pause. `Ctrl+Enter` translates immediately. Follow the CPU voices setup below to enable speech.

`requirements-lock.txt` records the exact Windows/CUDA environment used for verification (including test dependencies). To reproduce it, use `pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128` inside the environment.

For UI work without downloading or loading models:

```powershell
.\start.ps1 -Preview
```

## Share from your laptop

For a few people, use the included **Cloudflare Quick Tunnel**. No account, domain
purchase, email login or router port forwarding is needed. The site is still called
**Kath Bakwaas**; its temporary address looks like `https://random-words.trycloudflare.com`.
Anyone with the link can use it, including someone who receives a forwarded link.
It is an unlisted public site, not authenticated private access.

1. Install the small sharing dependency once:
   `.\.venv\Scripts\python.exe -m pip install -r requirements-sharing.txt`.
2. Run `start.cmd` and wait for **Ready** at `http://127.0.0.1:8000`.
3. Double-click `share.cmd`. It downloads a pinned Windows x64 `cloudflared` release
   from Cloudflare's GitHub repository, verifies its SHA-256, starts the gateway,
   and prints the link. Send that link to your friends.
4. Keep the laptop awake, online and plugged in, and keep both windows open.
   Use **Ctrl+C in the sharing window** or double-click **`stop-sharing.cmd`** to
   remove public access. Local translation stays running. Stop its own window
   separately when finished.

The current link is also saved in `.hosting/url.txt` while sharing runs. A restart
creates a different link. Gateway and tunnel diagnostics live in `.hosting/`;
these files and the downloaded binary in `.tools/` are ignored by Git. No system
service, firewall rule, startup entry or power setting is changed.

On Windows, sharing automatically requests that the laptop **stay awake while
plugged in**. The screen can turn off and Windows can lock. Stopping sharing
releases the request; unplugging restores normal battery sleep behavior. Keep
the lid open: closing it, explicitly choosing Sleep, hibernation, shutdown or
restarts can still take the site offline. This uses Windows' built-in
`SetThreadExecutionState` API, with no extra software or simulated mouse activity.
For an already-running session started before this feature, `keep-awake.cmd`
attaches the same helper without restarting the tunnel or changing the URL. That
helper exits after 12 hours, or earlier when sharing stops. Temporary gateway
failures do not end sleep prevention. To choose a different duration, run
`.\.venv\Scripts\python.exe scripts/keep_awake.py --hours 12`.
For an exact deadline, use `--until 2026-10-20T23:00:00-04:00` instead of
`--hours` (11 p.m. Toronto time on October 20 in this example). An explicit
timezone offset is required; the deadline must be within the next seven days.
Its live verification and expiry time are saved in `.hosting/keep-awake-status.json`.
Expiry only releases the helper's sleep-prevention request; it does not shut down
the website or change Windows' sleep settings. New `share.cmd` sessions keep their
own request active until sharing ends.

Traffic follows **browser → Cloudflare HTTPS → outbound tunnel → loopback gateway
on port 8010 → local model service on port 8000**. Both local ports bind only to
`127.0.0.1`. Cloudflare terminates HTTPS, so it is part of the trusted traffic path;
inference runs on your laptop. The gateway exposes only the page, static assets,
health, translation, prefill and speech routes. It does not expose FastAPI's schema
or arbitrary local services. It forwards no visitor credentials to the backend.

The gateway caps request bodies at 16 KiB, body uploads at five seconds, total
in-flight requests at 32, translation requests at three (one running plus up to
two waiting), prefill at one, and speech at one. All POSTs share a token bucket
of four requests/second with a burst of 24; this includes speculative typing
requests. Excess requests receive 429, rather than adding more model workers.
The existing GPU lock and memory checks remain in effect. Model weights,
quantization, decoding and diacritization are unchanged. Global limits prevent
visitors bypassing limits by supplying fake IP headers, but a determined visitor
could still monopolize the allowance. This is a small trial, not a full abuse
prevention service.

The page shows an overload message for rate limits or GPU memory pressure and
an unavailable/overloaded message for server and proxy failures, including HTML
error responses. After 15 seconds it shows that the server is still trying;
timeouts offer a retry instead of blaming the sentence length. Refresh existing
browser tabs after a frontend update to load the latest messages.

Cross-site browser POSTs are rejected. `robots.txt` and `X-Robots-Tag` ask search
engines not to index the site; these are requests to crawlers, not access controls.
Responses use `Referrer-Policy: no-referrer` and `Cache-Control: no-store`. The
application's existing content security policy is preserved. Do not post the link
in a public directory if you want to keep the audience small.

**Capacity:** start with two or three active typists and a handful of occasional
visitors. This is an estimate, not a concurrent-user benchmark. The 400-input audit
measured a 1.23-second median for fresh prompts and about 11.6 seconds at P95;
typing causes additional work and longer passages block the same single GPU
worker. At 1.23 seconds per job, a hypothetical stream of equally short jobs would
top out near 49/minute before typing work and network overhead. This is not a
sustained throughput claim. More people can open the page than can translate
simultaneously. CPU speech also serves one request at a time.

[Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
are free temporary demos with no uptime guarantee. For a permanent address later,
buy a domain and configure a named Cloudflare Tunnel, for example
`translate.yourdomain.com`; that example is not a reserved or purchased address.
The same laptop backend can remain in place. A permanent domain does not make
the laptop faster or keep it online when asleep.

## Local traffic counts

Double-click **`traffic.cmd`** for page loads, approximate visitors, successful
translations (fresh versus cached), and translation errors. Or run
`.\.venv\Scripts\python.exe scripts/traffic.py --json` for structured output.
Counts begin when the analytics-enabled sharing gateway starts; earlier traffic
cannot be reconstructed. They persist across restarts in `.hosting/traffic.sqlite3`.
There is no public analytics endpoint.

The gateway counts successful `GET /` responses as page loads and successful
`POST /api/translate` responses as translations. These are server responses, not
proof someone read the result. Typing may produce several successful translations
for one sentence. Static assets, health polls, speculative prefills and the
browser-only curated Kashmir example are not counted as translations. Busy,
cancelled and failed translation requests are reported separately.

Approximate visitors are distinct Cloudflare visitor IPs seen on successful page
or translation responses. People sharing Wi-Fi may count as one; changing IPs,
VPNs, link previews and bots can affect the count. No fingerprinting cookies or
browser identifiers are added. The gateway uses `CF-Connecting-IP` only from its
loopback tunnel connection, with proxy-header rewriting disabled, and hashes the
address using HMAC-SHA256 with a local secret before storing it. Raw IPs, submitted
English, Kashmiri output, query strings and user-agent strings are not stored in
analytics. Hashes are pseudonymous identifiers, not a guarantee of anonymity.

Aggregate counters and unique-IP fingerprints persist locally. Detailed timestamp,
request-category, response-status and elapsed-time records are limited to the most
recent seven days and 10,000 events, pruned when traffic is recorded. Database and
key files stay inside Git-ignored `.hosting/`. Locally marked verification requests
are separate from normal traffic; `scripts/traffic.py --checks` shows them.
Database write failures are logged and do not fail a translation, so such failures
can cause undercounting.

`scripts/deploy_gateway.py --prepare` records exact identities of the original
gateway supervisor and its Cloudflare child. After testing a staging gateway,
`--deploy` checks those identities and source hashes again, replaces only the
gateway supervisor and gateway, and retains the existing Cloudflare process and
URL. This Windows-only handoff causes a brief gateway interruption; model
inference settings and the model process remain unchanged. The replacement
supervisor still responds to `stop-sharing.cmd`. Its status is in
`.hosting/gateway-supervisor.json`. Do not restart `share.cmd` to deploy a gateway
update when a temporary link must be preserved.

## Model pipeline

### CPU voices (Bolbosh)

The **Male voice** and **Female voice** buttons play the Kashmiri translation.
During preparation the active button shows **Loading…**, and both voice buttons are disabled so accidental double-clicks do not cancel the request. Playback starts automatically. Click the active button while it is playing to stop; changing the English text stops old audio.
Speech uses the marked translation even when the diacritics display is switched off.

Install once in a separate environment, leaving translator dependencies unchanged:

```powershell
python -m venv .venv-tts
.\.venv-tts\Scripts\python.exe -m pip install -r requirements-tts.txt
.\.venv\Scripts\python.exe scripts/setup_bolbosh.py
```

Restart `start.cmd` after setup. Bolbosh preloads after the translator starts.
`requirements-tts-lock.txt` records the tested CPU environment; install it with
`--extra-index-url https://download.pytorch.org/whl/cpu` to reproduce exact versions.
It uses **CPU-only PyTorch**, two CPU threads and a separate process: no CUDA
context or GPU memory. The acoustic model has 20.9M parameters and HiFi-GAN
has 13.9M. Model files download about 307 MB. The checkpoint is pinned to
`718203e01e0b4ddab3e925445619c69cddeb7231` from
[GAASH-Lab/Matcha-TTS-Kashmiri](https://huggingface.co/GAASH-Lab/Matcha-TTS-Kashmiri).
It uses the official demo's 10-step, temperature 0.667, rate 1.0 settings and
speaker IDs 422 (male) / 423 (female). No translation settings are changed.

Only one speech job runs at once. Audio is cached in memory (32 entries / 16 MB),
and long passages are split at sentences/words. Repeated playback is cached.
Six real local samples took 0.84–3.06 seconds to synthesize after a 5.1-second
initial load; longer passages take longer. This is a runtime check, not a
human assessment of pronunciation. Run `python scripts/check_bolbosh.py` with
the translator environment for WAV samples and timings in `artifacts/`.

Bolbosh, HiFi-GAN and KashmiriNormalizer source licenses are retained under
`vendor/bolbosh/`. See its README for the pinned sources and inference-only changes.

### Translation

1. [Jibby1729/Koshur-Sarvam-4B](https://huggingface.co/Jibby1729/Koshur-Sarvam-4B), pinned to `09ca1350c5654eb05fc585787bbe0f550daafff5`. The exact released system prompt, deterministic four-beam decoding, length penalty 0.7, and maximum 512 generated tokens. NF4 quantization with double quantization and BF16 computation (FP16 on GPUs without BF16). Generation caching is enabled for latency.
2. [Jibby1729/Koshur-JibDiacritizer-152M](https://huggingface.co/Jibby1729/Koshur-JibDiacritizer-152M), pinned to `fd5df75a44cf09889c4cd373a6156f0cc2f92596`. Uses the bundled, reviewed `KashmiriRestorer` implementation, default IndicVoices style and no-mark bias. A letter-preservation check compares the normalized draft and restored output after stripping marks.

Both marked and original draft text are returned together, so switching diacritics does not run another translation. Beam search returns the completed translation after a typing pause; it does not stream provisional tokens. Longer passages take longer than short phrases. A generation that reaches the token cap without finishing is reported as incomplete, rather than displayed as a finished translation.

While typing, the page sends optional `/api/prefill` requests. The server runs the normal `generate()` preparation and its complete four-beam prompt forward, then stops before selecting a generated token. It stores an independent snapshot of the prompt's key/value tensors and next-token logits. Translation reuses that snapshot only if every token of the complete input matches. Appending, deleting, or editing words requires a new complete prefill; partial prefixes are never reused. Generated Kashmiri never enters this cache. There are at most four page sessions with a two-minute TTL. Speculative prefill yields to translation work and falls back to ordinary inference if unavailable.

The earlier incremental prefix cache was rejected after broad regression testing found changed translations, including the user's reported hungry/annoyed sentence. Identical weights and beam settings alone did not preserve the outputs of that computation path. The replacement moves the same complete prompt computation earlier in time instead. This can hide input processing behind typing, but does not eliminate the cost of generating the Kashmiri output.

The page cancels obsolete fetches and ignores stale responses. The server monitors disconnects, stops an obsolete generation at the next decoding step, and serializes GPU work. It bounds pending requests to eight and caches the latest 128 successful passages in memory. Ctrl+Enter immediately translates the current passage. Input is limited to 500 characters to keep requests manageable on a laptop.

On the validated Windows/Ada GPU configuration, the decoder uses a native NF4 kernel and exact-length CUDA graphs. It preserves the original BF16 operands, sequential accumulation, four beams, and CPU FP32 diacritizer. All 238 quantized projections are checked against the original implementation during startup. A lossless weight-layout permutation keeps one compressed copy; prefill reconstructs the original bytes. At most 52 decoder positions are captured at startup, with two prefill graphs and four shapes per beam helper. Other positions run without a graph, and contexts beyond the 128-token buffer continue through the dynamic cache without shortening the input or output. Graph preparation adds startup time. On hybrid Windows CPUs, this process uses physical performance cores while retaining four CPU threads.

Before inference, the server checks for 1.5 GiB of free GPU memory. If necessary, it releases optional graphs and prompt caches. If memory is still too tight, it returns a retryable response before starting generation. This reduces OOM risk; it cannot prevent another application from allocating GPU memory concurrently. The conversion-cache fallback is limited to 1000 MiB and leaves 2 GiB before graph capture.

Other local GPU models share this budget. For example, OpenWhispr's GPU dictation
service used about 1.9 GiB on this laptop and left too little headroom for
translation. Temporarily stopping that model service restored translation.
Keep other GPU workloads within the available memory; the server deliberately
keeps its safety threshold instead of risking an OOM. Graph cleanup also releases
the projection output tensors that could otherwise retain retired allocator segments.

The native kernel requires the CUDA Toolkit's `nvcc` and Visual Studio C++ tools already installed locally; it builds into ignored `artifacts/native/`. No compiler is downloaded automatically. Unsupported configurations or setup failure use the earlier bounded conversion-cache/MLP-graph backend. Set `KOSHUR_NATIVE_DECODE=0` to select that backend, `KOSHUR_OPTIMIZE_EXECUTION=0` for stock inference, or `KOSHUR_PERFORMANCE_CORES=0` for normal Windows CPU scheduling. See [measurements and remaining work](INFERENCE_NOTES.md). Matching the stock translations and maintaining GPU headroom take priority over speed; the working latency target is around 1.5 seconds for ordinary sentences, with longer passages taking longer.

## API

- `GET /api/health`: `{ "state": "starting|downloading|loading|ready|error|offline", "message": "…" }`.
- `POST /api/translate` with `{ "text": "How are you?", "session_id": "page-uuid" }`: `{ "translation": "…", "draft": "…", "elapsed_seconds": 1.23, "cached": false, "timings": { ... } }`. `session_id` is optional; timings include generation, restoration, first decoding step, and reused prompt token count.
- `POST /api/prefill` with the same body and a required `session_id`: prepares the complete four-beam prompt for exact-input reuse. Returns `prefilled: false` when inference is busy; this speculative request never queues ahead of translation.
- `POST /api/speech` with `{ "text": "کٲشُر", "voice": "male" }` (or `female`): mono 22,050 Hz PCM WAV; max 1,500 characters, 429 when speech is busy. Speech failures do not discard the translation.
- 422 for invalid input/incomplete output, 503 while unavailable, 429 when the inference queue is full. Model startup and inference failures appear in the server terminal.

FastAPI serves the page and API from the same origin. No browser API keys or external inference service. Typed text goes to the local server and remains in its in-memory cache until eviction/restart; the application does not write passages to a database, local storage, or access logs. Only the bigger-text preference is stored in the browser. Hugging Face is contacted to download model files. Model caches and environments are ignored by Git.

The service binds to **127.0.0.1** by default. Optional laptop sharing uses the
bounded gateway described above. Serving the static files alone cannot run the
models. Use one Uvicorn model worker per GPU to avoid loading duplicate copies.
No paid infrastructure is provisioned by this project. A standard CPU VPS alone
cannot run this application's CUDA translator.

Cloud migration is not yet implemented or benchmarked. The fastest native decoder is currently specific to **Windows/Ada GPUs**; moving to Linux or another GPU requires choosing the compatible path and repeating output-equality and latency tests. Cold-starting a cloud model also adds loading time. Keep four-beam decoding and the CPU diacritizer when comparing deployments.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
node --check web/app.js
node tests/test_typing.mjs
node tests/test_speech.mjs
```

API tests use a stand-in engine to check validation, readiness, cache limits, error recovery, and cancellation without GPU downloads. They do not measure translation quality. For an actual model smoke test, start normally and submit an English sentence through the page or API.

The initial seven-case GPU smoke check missed a regression. The subsequent 161-input audit found that the old incremental typing cache changed 15 translations, including the user's reported example. That path has been removed. The native decoder with fresh prompts and the replacement complete-prompt cache each matched all 161 stock token sequences, drafts, and diacritized outputs. This is regression coverage, not a guarantee for all possible inputs or a human translation-quality score. See the latest section of [INFERENCE_NOTES.md](INFERENCE_NOTES.md) and the local `artifacts/accuracy-summary.md` report.

The expanded audit added **400 distinct inputs**, producing **1,200 translations** across stock inference, optimized fresh prompts, and optimized prepared prompts. Both optimized paths matched all 400 reference token sequences, drafts and diacritized outputs, with no errors or CUDA OOMs. The median translation times were 5.13 s, 1.23 s and 0.89 s respectively; prepared-prompt timing excludes preparation. Longer passages remained slower (optimized P95 about 11.6 s). These results apply to the tested hardware and software, not every device or possible input.

Python tests cover API validation, exact-input invalidation, isolation, cancellation,
memory-pressure handling, bounded CPU speech, and the sharing gateway's route,
body, origin, rate and concurrency limits. JavaScript checks cover typing, saved
examples, speech loading/double-click protection, replay, stopping, errors and
stale responses. Browser checks cover the page and voice playback. Six real CPU
speech samples passed WAV checks; pronunciation quality still needs human review.

With the app stopped, run `python scripts/benchmark_prefix.py` using the project environment for fresh-versus-cached four-beam comparisons. Results are written under the ignored `artifacts/` directory. `scripts/benchmark_latency.py` profiles the pipeline; its optional `--include-greedy` flag is an exploratory output comparison and never changes app settings. Production continues to use four beams and CPU FP32 diacritization.

For repeatable native-backend checks, stop the service and run `python scripts/create_native_reference.py`, then `python scripts/verify_native_backend.py`. These compare complete token sequences and both text outputs for 13 passages, typing-time prefix reuse, in-flight cancellation, and continuation past the graph cache limit. They perform new inference rather than looking up completed translations.

For the broader audit, run `scripts/compare_accuracy.py --backend stock`, then `--backend native`, then `--backend native --typing --report accuracy-native-complete-prompt.json --fail-fast`, using the project Python environment. Run `scripts/stress_accuracy_memory.py` for near-limit inputs, cancellation and controlled memory-pressure recovery. Keep only one GPU benchmark or web service running at a time. `scripts/summarize_accuracy.py` renders the saved comparisons into Markdown.

To reproduce the expanded 400-input audit, stop the local server, run `scripts/build_expanded_corpus.py`, then `scripts/run_expanded_accuracy.py`, then `scripts/summarize_expanded_accuracy.py` using `.venv/Scripts/python.exe`. The runner restores the web server afterwards. Raw outputs, timing reports, WAV samples and generated native binaries stay in the ignored `artifacts/` directory; do not start a second GPU model beside the web service.

## Repository layout

| Path | Contents |
| --- | --- |
| `web/` | Page, styles, translation and audio controls, local font |
| `app/` | FastAPI API, model pipeline, bounded caches and CPU speech worker |
| `app/kernels/` | Native NF4 decoder source |
| `vendor/bolbosh/` | Pinned inference sources and upstream license notices |
| `tests/` | Backend and JavaScript regression checks |
| `scripts/` | Setup, reproducible comparisons, and earlier experimental benchmarks |
| `INFERENCE_NOTES.md` | Optimization history, accepted/rejected paths and measurements |

Experimental scripts are not used by the web application. `.venv/`, `.venv-tts/`, `.model-cache/`, `artifacts/`, credentials and compiled binaries are excluded from Git. Model weights download during setup rather than being committed.

## Attribution

The translator weights inherit **GPL-3.0 and the Gemma terms**; the diacritizer and its loader are **Apache-2.0**. See the linked model repositories for their licenses and notices. Model weights are downloaded from those repositories, not redistributed here. Noto Naskh Arabic is bundled under the [SIL Open Font License](web/fonts/OFL.txt), from [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Naskh+Arabic).
