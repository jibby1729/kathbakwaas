"""The released two-stage Koshur pipeline, with cooperative cancellation."""

import importlib.util
import logging
import os
from pathlib import Path
from threading import Event
from time import perf_counter

from .prefix_cache import PromptCache

TRANSLATOR = "Jibby1729/Koshur-Sarvam-4B"
RESTORER = "Jibby1729/Koshur-JibDiacritizer-152M"
# Pin the weights AND the executable restorer code reviewed for this application.
TRANSLATOR_REVISION = "09ca1350c5654eb05fc585787bbe0f550daafff5"
RESTORER_REVISION = "fd5df75a44cf09889c4cd373a6156f0cc2f92596"
logger = logging.getLogger(__name__)


class TranslationCancelled(Exception):
    pass


class TranslationEngine:
    def __init__(self):
        self.state = "starting"
        self.message = "Getting the translator ready…"
        self.error = None
        self.prompt_cache = PromptCache()

    def status(self):
        return {"state": self.state, "message": self.message}

    def load(self):
        try:
            # Direct HTTPS downloads are more predictable on Windows than Xet.
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA-enabled NVIDIA GPU is required. Install the CUDA PyTorch build; see README.md.")
            # Avoid letting the small CPU restorer monopolize every logical core.
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            cache_root = Path(os.getenv(
                "HF_HOME", str(Path(__file__).resolve().parent.parent / ".model-cache" / "huggingface"),
            )) / "hub"
            self.state, self.message = "downloading", "Downloading the language models. The first start takes a little longer…"
            translator_dir = snapshot_download(
                TRANSLATOR, revision=TRANSLATOR_REVISION,
                allow_patterns=["*.json", "*.safetensors", "*.jinja"],
                cache_dir=cache_root,
                max_workers=2,
            )
            restorer_dir = snapshot_download(
                RESTORER, revision=RESTORER_REVISION,
                cache_dir=cache_root,
                max_workers=2,
            )
            self.state, self.message = "loading", "Warming up the translator…"
            self.tokenizer = AutoTokenizer.from_pretrained(translator_dir, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(
                translator_dir,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
                ),
                dtype=compute_dtype, device_map={"": 0},
                attn_implementation="sdpa",
            ).eval()
            spec = importlib.util.spec_from_file_location("koshur_pinned_restorer", Path(restorer_dir) / "kashmiri_restorer.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.strip_marks = module.strip_marks
            self.restorer = module.KashmiriRestorer.from_pretrained(restorer_dir, device="cpu")
            self.execution_optimization = None
            if os.getenv("KOSHUR_OPTIMIZE_EXECUTION", "1") != "0":
                from .fast_execution import optimize_fast_execution
                from .cpu_execution import prefer_performance_cores
                try:
                    self.cpu_execution = prefer_performance_cores()
                    if self.cpu_execution:
                        logger.info('Using physical performance cores for this process: %s', self.cpu_execution['logical_processors'])
                except Exception:
                    logger.exception('Optional CPU placement unavailable; retaining system scheduling')
                self.fast_execution = optimize_fast_execution(self.model)
                if self.fast_execution is not None:
                    self.execution_optimization = {
                        'backend': 'native_nf4', 'validated_projections': 238,
                        'decoder_graphs': len(self.fast_execution['decoder'].graphs),
                        'graph_capacity': self.fast_execution['graph_capacity'],
                    }
                else:
                    from .execution import optimize_execution
                    self.execution_optimization = optimize_execution(self.model)
                # Touch the CPU model and initialize its kernels before requests.
                # Keep FP32 and the same four-thread numerical execution path.
                with torch.inference_mode():
                    self.restorer.model(torch.ones((1, 128), dtype=torch.long))
            self.state, self.message = "ready", "Ready to translate"
            logger.info("Koshur pipeline ready: NF4 translator on CUDA; restorer on CPU")
        except Exception as exc:
            self.state, self.message = "error", "The translator could not start. Please check the server and try again."
            self.error = str(exc)
            logger.exception("Model initialization failed")

    def encode(self, text):
        return self.tokenizer.apply_chat_template([
            {"role": "system", "content": "Translate the text below to Kashmiri."},
            {"role": "user", "content": text},
        ], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)

    def prefill(self, text, session):
        import torch
        from copy import deepcopy
        from transformers.modeling_outputs import CausalLMOutputWithPast
        from .gpu_memory import ensure_headroom

        ensure_headroom(self)
        started = perf_counter()
        encoded = self.encode(text)
        ids = encoded["input_ids"][0].tolist()
        entry = self.prompt_cache.get(session)
        if entry and tuple(ids) == entry[0]:
            return {"prefilled": True, "reused_tokens": len(ids), "cached_tokens": len(ids)}

        class PrefillFinished(Exception):
            pass

        # Use generate() itself to prepare the same four-beam inputs, positions
        # and model call as a fresh translation, then stop before token selection.
        original_prefill = self.model._prefill
        saved = None

        def capture(*args, **kwargs):
            nonlocal saved
            output = original_prefill(*args, **kwargs)
            cache = output.past_key_values
            if hasattr(cache, 'export_to_dynamic'):
                cache = cache.export_to_dynamic(self.model)
            else:
                cache = deepcopy(cache)
            saved = CausalLMOutputWithPast(logits=output.logits.clone(), past_key_values=cache)
            raise PrefillFinished()

        self.prompt_cache.discard(session)
        del entry
        with torch.inference_mode():
            self.model._prefill = capture
            try:
                self.model.generate(**encoded, **self.generation_options())
            except PrefillFinished:
                self.prompt_cache.put(session, ids, saved)
            finally:
                self.model._prefill = original_prefill
            # The snapshot is ready when this request finishes, and its timing
            # includes GPU execution rather than only asynchronous dispatch.
            torch.cuda.current_stream().synchronize()
        return {"prefilled": saved is not None, "reused_tokens": 0, "cached_tokens": len(ids),
                "prefill_ms": round((perf_counter() - started) * 1000, 2)}

    def generation_options(self):
        return dict(num_beams=4, length_penalty=0.7, early_stopping=True,
                    do_sample=False, max_new_tokens=512, use_cache=True,
                    eos_token_id=list(dict.fromkeys([
                        self.tokenizer.convert_tokens_to_ids('<end_of_turn>'), self.tokenizer.eos_token_id,
                    ])))

    def translate(self, text: str, cancelled: Event, session=None):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        from .gpu_memory import ensure_headroom

        class StopOnCancel(StoppingCriteria):
            first_step = None
            first_step_event = None

            def __call__(self, input_ids, scores, **kwargs):
                if self.first_step is None:
                    self.first_step = perf_counter()
                    self.first_step_event = torch.cuda.Event(enable_timing=True)
                    self.first_step_event.record()
                return cancelled.is_set()

        def check_cancelled():
            if cancelled.is_set():
                raise TranslationCancelled()

        check_cancelled()
        ensure_headroom(self)
        started = perf_counter()
        encoded = self.encode(text)
        end_ids = list(dict.fromkeys([
            self.tokenizer.convert_tokens_to_ids("<end_of_turn>"), self.tokenizer.eos_token_id,
        ]))
        stopper = StopOnCancel()
        reused = 0
        with torch.inference_mode():
            prepared = None
            if session:
                try:
                    prepared = self.prompt_cache.fork(session, encoded["input_ids"][0].tolist())
                except Exception:
                    self.prompt_cache.discard(session)
                    logger.exception("Prompt cache unavailable; using a full prefill")
            generation_started = perf_counter()
            generation_start_event = torch.cuda.Event(enable_timing=True)
            generation_start_event.record()
            original_prefill = self.model._prefill

            def use_prepared(input_ids, generation_config, model_kwargs, is_first_iteration=True):
                nonlocal reused
                if (prepared is not None and generation_config.num_beams == 4
                        and input_ids.shape == (4, encoded['input_ids'].shape[1])
                        and torch.equal(input_ids, encoded['input_ids'].expand(4, -1))
                        and model_kwargs['past_key_values'].get_seq_length() == 0):
                    reused = input_ids.shape[1]
                    return prepared
                return original_prefill(input_ids, generation_config, model_kwargs, is_first_iteration)

            if prepared is not None:
                self.model._prefill = use_prepared
            try:
                output = self.model.generate(
                    **encoded, **self.generation_options(),
                    stopping_criteria=StoppingCriteriaList([stopper]),
                )
            finally:
                self.model._prefill = original_prefill
        generation_finished = perf_counter()
        check_cancelled()
        # Graph replay is asynchronous. A host timestamp at the first stopping
        # check measures dispatch, not completion of the first GPU decoding step.
        first_step_ms = (generation_start_event.elapsed_time(stopper.first_step_event)
                         if stopper.first_step_event is not None
                         else (generation_finished - generation_started) * 1000)
        generated = output[0, encoded["input_ids"].shape[1]:]
        # Never present a token-limit fragment as a completed translation.
        if generated.numel() == 512 and generated[-1].item() not in end_ids:
            raise ValueError("This translation was too long to finish. Please try a shorter sentence.")
        draft = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not draft:
            raise ValueError("We couldn’t translate that text. Please try a complete sentence.")
        restored = self.restorer.restore([draft])[0]
        restored_at = perf_counter()
        check_cancelled()
        if self.strip_marks(restored) != self.strip_marks(draft):
            raise RuntimeError("The diacritizer changed the underlying text.")
        return {
            "translation": restored, "draft": draft,
            "elapsed_seconds": round(perf_counter() - started, 2),
            "timings": {
                "generate_ms": round((generation_finished - generation_started) * 1000, 2),
                "diacritizer_ms": round((restored_at - generation_finished) * 1000, 2),
                "first_step_ms": round(first_step_ms, 2),
                "reused_prompt_tokens": reused,
                "output_tokens": len(generated),
            },
        }
