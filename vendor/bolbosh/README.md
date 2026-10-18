# Bolbosh inference sources

Source: https://github.com/gaash-lab/Bolbosh
Revision: `7435b77029a691cc00a335356013ca6ee78f794a` (MIT; LICENSE included).

KashmiriNormalizer: https://github.com/abdulmuizz0903/KashmiriNormalizer
Revision: `9bb966e76b88caf875bfb9facf3036bef706427e` (MIT; license included).

This is an inference-only subset. The encoder, flow decoder, transformer,
vocoder, text symbols and normalizer retain upstream computation. Training
forward and Lightning logging/optimizer dependencies are omitted; the base
class registers the same mel statistics. Plotting imports are lazy. A 2,600
mel-frame limit rejects oversized segments before decoder allocation.

CPU FP32, 10 Euler steps, temperature 0.667, speaking rate 1.0 and speaker IDs
422/423 follow the official demo's default speech settings. Translation is
handled separately by this application's existing Koshur models.

`scripts/setup_bolbosh.py` downloads the pinned checkpoint and original vocoder.
Weights remain in the ignored `.model-cache/bolbosh/` directory.
