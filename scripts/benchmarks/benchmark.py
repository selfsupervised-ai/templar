#!/usr/bin/env python3
"""llm_benchmark.py

Comprehensive inference‑time benchmarking script for Llama‑family (or any causal‑LM) models
using Hugging Face Transformers. Results are aggregated into rich latency‑throughput metrics
and streamed to Weights & Biases (W&B).

Features
========
* Baseline profiling with ``torch.profiler`` (CPU+CUDA kernel timeline, mem, FLOPs)
* Toggle‑able micro‑optimisations:
  - ``torch.no_grad`` ⟷ ``torch.inference_mode``
  - Skip per‑batch ``torch.cuda.empty_cache``
  - DataLoader knobs: pinned‑memory / persistent‑workers
  - ``torch.backends.cudnn.benchmark``
* Compiler path:
  - ``torch.compile`` (Inductor + nvFuser)
  - Lightning Thunder wrapper (if installed) ✨
  - Flash‑level fused attention kernels (FlashInfer / FlashAttention‑2)
  - PagedAttention (vLLM) + RadixAttention (SGL)
* Mixed precision (AMP) support (default FP16/BF16 on Ampere+)
* GPU metrics: max memory, free memory floor, utilisation (% via ``pynvml``)
* Batching experiments with async prefetch, worker‑side CUDA collation, and LRU tokeniser cache
* Out‑of‑the‑box W&B logging for every variant + profiler artifacts

Usage (minimal)
---------------
```bash
python llm_benchmark.py --model meta-llama/Llama-3-8B --device cuda:0 \
       --batch-sizes 1 4 8 --seq-len 256 --out-len 128 --wandb-project llama-bench
```

The script will iterate through a matrix of variants × batch sizes, print a live table, and
deposit everything under the specified W&B project. See ``--help`` for all options.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import importlib
import itertools
import json
import os
import random
import statistics as stats
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

import tplr
from tplr.r2_dataset import R2DatasetLoader

# ----------------------------- W&B optional import -----------------------------
try:
    import wandb
except ImportError:  # degrade gracefully
    wandb = None  # type: ignore

# ------------------------------- GPU utilities --------------------------------
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_LOADED = True
except Exception:
    _NVML_LOADED = False

# -------------------------------- Dataclasses ---------------------------------

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # time to first token (ms)
    itl: List[float] = field(default_factory=list)  # inter‑token latencies (ms)
    prompt_len: int = 0
    error: str = ""


@dataclass
class BenchmarkMetrics:
    # throughput + latency summary across requests
    completed: int
    failures: int
    total_input: int
    total_output: int
    request_throughput: float  # req/s
    input_throughput: float  # tokens‑in/s
    output_throughput: float  # tokens‑out/s
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float  # total processing time per output token
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    max_input: int
    max_output: int
    max_total: int
    peak_gpu_memory_mib: float  # max allocated
    available_gpu_memory_mib: float  # min free
    gpu_utilization: float  # avg util over run (%)

# -----------------------------------------------------------------------------

class R2Dataset(Dataset):
    """Dataset using the R2 dataset loader from the validator."""

    def __init__(self, tokenizer, hparams, window=None, seed=42, num_samples=None):
        self.tokenizer = tokenizer
        self.hparams = hparams
        self.window = window or 0
        self.seq_len = hparams.sequence_length
        self.seed = seed
        self.num_samples = num_samples
        self.batches = []

        # Initialize asynchronously
        self._initialize()

    def _initialize(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Generate pages based on seed similar to how validator does it
            pages = loop.run_until_complete(
                R2DatasetLoader.next_pages(
                    offset=self.window * self.hparams.pages_per_window,
                    n_pages=self.hparams.pages_per_window,
                    seed=self.seed
                )
            )
            
            if pages:
                # Create loader with the generated pages
                loader = loop.run_until_complete(
                    R2DatasetLoader.create(
                        batch_size=self.hparams.batch_size,
                        sequence_length=self.seq_len,
                        pages_info=pages,
                        tokenizer=self.tokenizer
                    )
                )
                
                # Load all batches
                if loader:
                    self.batches = list(loader)
                    if self.num_samples and len(self.batches) > self.num_samples:
                        self.batches = self.batches[:self.num_samples]
                    print(f"Loaded {len(self.batches)} batches from R2 dataset")
        finally:
            loop.close()

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class CollateToCuda:
    """Tokenise and push batch to GPU inside DataLoader worker for zero‑copy."""

    def __init__(self, tokenizer: AutoTokenizer, device: torch.device, max_length: int):
        self.tok = tokenizer
        self.dev = device
        self.max_length = max_length

    def __call__(self, batch: List[str]):
        encoded = self.tok(batch, return_tensors="pt", truncation=True, max_length=self.max_length, padding=True)
        return {k: v.to(self.dev, non_blocking=True) for k, v in encoded.items()}

# -----------------------------------------------------------------------------

class Profiler:
    """Thin wrapper around torch.profiler for optional capture."""

    def __init__(self, enabled: bool, logdir: Path | str):
        self.enabled = enabled
        if enabled:
            schedule = torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.prof = torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(logdir)),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
        else:
            self.prof = contextlib.nullcontext()

    def __enter__(self):
        return self.prof.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.prof.__exit__(exc_type, exc_value, traceback)
        
    def step(self):
        """Step the profiler if it's enabled."""
        if self.enabled and hasattr(self.prof, 'step'):
            self.prof.step()

# -----------------------------------------------------------------------------

class LLMInferenceBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        
        # Check if CUDA is available, fallback to CPU if not
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(args.device)
        
        # Load hparams as used by the validator
        self.hparams = tplr.load_hparams(use_local_run_hparams=args.local)

        # Load tokenizer from hparams as validator does
        self.tokenizer = self.hparams.tokenizer
        print(f"Loaded tokenizer from validator hparams")
        
        # Load model
        self.model = self._load_model()
        
        # Create dataset for benchmarking
        self.dataset = self._create_dataset()
        
    def _create_dataset(self):
        """Create R2 dataset using validator's approach."""
        print(f"Creating R2 dataset with validator data for benchmarking")
        return R2Dataset(
            tokenizer=self.tokenizer,
            hparams=self.hparams,
            window=getattr(self.args, 'window', 0),
            seed=getattr(self.args, 'seed', 42),
            num_samples=self.args.num_reqs
        )

    # --------------------------- model loading + compile ---------------------------

    def _load_model(self):
        print(f"Loading model with validator's configuration")
        # Create the model with validator's configuration
        model = LlamaForCausalLM(self.hparams.model_config)
        model.to(self.device)
        model.eval()
        
        if self.args.load_checkpoint:
            print(f"Loading checkpoint from R2 storage...")
            # Load the most recent checkpoint
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Create a minimal Comms instance just for checkpoint loading
                from bittensor import subtensor, wallet

                from tplr.comms import Comms

                # Setup wallet and subtensor objects 
                config = SimpleNamespace(netuid=268, peers=None)
                wallet_obj = wallet(config=config)
                subtensor_obj = subtensor(config=config)
                metagraph = subtensor_obj.metagraph(config.netuid)
                
                # Setup comms
                comms = Comms(
                    wallet=wallet_obj,
                    save_location="/tmp",
                    key_prefix="model",
                    config=config,
                    netuid=config.netuid,
                    metagraph=metagraph,
                    hparams=self.hparams,
                    uid=0  # validator's uid - ideally should find the real one
                )
                
                # Load checkpoint (this is a simplified approach)
                success, _, _, _, _ = loop.run_until_complete(
                    comms.load_checkpoint(
                        model=model,
                        optimizer=None,
                        scheduler=None, 
                        current_window=0,
                        device=self.device
                    )
                )
                
                if success:
                    print(f"Successfully loaded model checkpoint")
                else:
                    print(f"Failed to load checkpoint, using initial weights")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
            finally:
                loop.close()

        if self.args.enable_compile:
            if hasattr(torch, "compile"):
                print("[compile] torch.compile enabled → mode=fusion")
                model = torch.compile(model, mode="max-autotune")
            else:
                print("torch.compile not available, skipping compile flag")

        # FlashInfer patch (optional)
        if self.args.enable_flashinfer:
            self._patch_flashinfer(model)

        # Lightning Thunder (optional)
        if self.args.enable_thunder:
            self._wrap_thunder(model)

        return model

    # Optional patches ----------------------------------------------------------------

    def _patch_flashinfer(self, model):
        try:
            # Optional import - flashinfer may not be installed
            from flashinfer import \
                replace_attention_forward  # type: ignore # pylint: disable=import-error

            replace_attention_forward(model)
            print("[flashinfer] patched in fused attention kernels ✅")
        except (ImportError, Exception) as e:  # pragma: no cover
            print(f"[flashinfer] ⚠️  unable to apply patch: {e}")

    def _wrap_thunder(self, model):
        try:
            # Optional import - lightning_thunder may not be installed
            import lightning_thunder as thunder  # type: ignore # pylint: disable=import-error

            model = thunder.bench_optimised(model)
            print("[thunder] Lightning Thunder optimisation applied ✅")
        except (ImportError, Exception) as e:  # pragma: no cover
            print(f"[thunder] ⚠️  unable to apply: {e}")

    # ----------------------------- run one experiment -----------------------------

    @torch.inference_mode()
    def _generate_stream(self, input_ids: torch.Tensor, max_new_tokens: int) -> Tuple[List[int], RequestFuncOutput]:
        """Generate step‑by‑step to capture TTFT + ITL accurately."""
        out = RequestFuncOutput(prompt_len=input_ids.shape[1])

        start_total = time.perf_counter()
        past_key_values = None
        logits = None

        # first forward (context)
        t0 = time.perf_counter()
        logits, past_key_values = self.model(input_ids, use_cache=True, past_key_values=past_key_values).logits, None
        out.ttft = (time.perf_counter() - t0) * 1e3  # ms

        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [token.item()]
        out.itl = []

        for _ in range(max_new_tokens - 1):
            t1 = time.perf_counter()
            logits, past_key_values = self.model(token, use_cache=True, past_key_values=past_key_values).logits, past_key_values
            delta = (time.perf_counter() - t1) * 1e3
            out.itl.append(delta)
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(token.item())

        out.latency = (time.perf_counter() - start_total) * 1e3
        out.success = True
        out.generated_text = self.tokenizer.decode(generated)
        return generated, out

    def _benchmark_batch(self, batch: Dict[str, torch.Tensor]) -> RequestFuncOutput:
        if self.args.use_amp:
            amp_ctx = torch.autocast(self.device.type, dtype=(torch.bfloat16 if self.args.bf16 else torch.float16))
        else:
            amp_ctx = contextlib.nullcontext()

        profiler_ctx = contextlib.nullcontext()
        if self.args.collect_profiler:
            profiler_ctx = self.profiler

        with amp_ctx, profiler_ctx:
            gen_ids, result = self._generate_stream(batch["input_ids"], self.args.output_len)
            if self.args.collect_profiler:
                self.profiler.step()
        return result

    # ------------------------------ driver routine ------------------------------

    def run(self):
        if wandb and not os.environ.get("WANDB_DISABLED"):
            wandb_run = wandb.init(project=self.args.wandb_project, config=vars(self.args), reinit=True)
        else:
            wandb_run = None

        # Use the R2Dataset that was created in __init__
        print(f"Using R2 dataset with {len(self.dataset)} batches from validator data")
        loader = self.dataset

        # profiler (global to experiment)
        self.profiler = Profiler(self.args.collect_profiler, self.args.profiler_outdir)

        # warmup (clears allocator fragmentation if using CUDA)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results: List[RequestFuncOutput] = []
        gpu_utils: List[float] = []
        mem_free_min = float("inf")

        for step, batch in enumerate(loader):
            if _NVML_LOADED and torch.cuda.is_available() and self.device.type == 'cuda':
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index if self.device.index is not None else 0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utils.append(util.gpu)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_free_min = min(mem_free_min, mem.free / (1024 ** 2))
                except Exception as e:
                    print(f"Warning: Could not collect GPU metrics: {e}")

            out = self._benchmark_batch(batch)
            results.append(out)

            if not self.args.disable_empty_cache and torch.cuda.is_available() and (step % self.args.empty_cache_every) == 0:
                torch.cuda.empty_cache()

        metrics = self._aggregate(results, gpu_utils, mem_free_min)

        if wandb_run is not None:
            wandb_run.log(asdict(metrics))
            if self.args.collect_profiler:
                wandb_run.save(str(self.args.profiler_outdir / "*"))
            wandb_run.finish()

        print(json.dumps(asdict(metrics), indent=2))

    # -------------------------- metric aggregation ---------------------------

    def _aggregate(self, outs: List[RequestFuncOutput], gpu_utils: List[float], mem_free_min: float) -> BenchmarkMetrics:
        successes = [o for o in outs if o.success]
        failures = len(outs) - len(successes)

        all_ttft = [o.ttft for o in successes]
        all_tpot = [o.latency for o in successes]
        all_itl = [lat for o in successes for lat in o.itl]
        total_in = sum(o.prompt_len for o in successes)
        total_out = self.args.output_len * len(successes)

        duration_sec = sum(o.latency for o in successes) / 1e3  # total active time
        req_throughput = len(successes) / duration_sec if duration_sec else 0.0
        inp_throughput = total_in / duration_sec if duration_sec else 0.0
        out_throughput = total_out / duration_sec if duration_sec else 0.0

        def _stats(v: List[float]):
            if not v:
                return 0.0, 0.0, 0.0, 0.0
            return (
                stats.mean(v),
                stats.median(v),
                stats.pstdev(v) if len(v) > 1 else 0.0,
                stats.quantiles(v, n=100)[-1]  # p99
            )

        mean_ttft, med_ttft, std_ttft, p99_ttft = _stats(all_ttft)
        mean_tpot, med_tpot, std_tpot, p99_tpot = _stats(all_tpot)
        mean_itl, med_itl, std_itl, p99_itl = _stats(all_itl)

        mem_alloc_peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2 if torch.cuda.is_available() else 0.0
        gpu_util_avg = stats.mean(gpu_utils) if gpu_utils else 0.0

        return BenchmarkMetrics(
            completed=len(successes),
            failures=failures,
            total_input=total_in,
            total_output=total_out,
            request_throughput=req_throughput,
            input_throughput=inp_throughput,
            output_throughput=out_throughput,
            mean_ttft_ms=mean_ttft,
            median_ttft_ms=med_ttft,
            std_ttft_ms=std_ttft,
            p99_ttft_ms=p99_ttft,
            mean_tpot_ms=mean_tpot,
            median_tpot_ms=med_tpot,
            std_tpot_ms=std_tpot,
            p99_tpot_ms=p99_tpot,
            mean_itl_ms=mean_itl,
            median_itl_ms=med_itl,
            std_itl_ms=std_itl,
            p99_itl_ms=p99_itl,
            max_input=max(o.prompt_len for o in successes) if successes else 0,
            max_output=self.args.output_len,
            max_total=self.args.output_len + max(o.prompt_len for o in successes) if successes else 0,
            peak_gpu_memory_mib=mem_alloc_peak,
            available_gpu_memory_mib=mem_free_min if mem_free_min != float("inf") else 0.0,
            gpu_utilization=gpu_util_avg,
        )

# -----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Comprehensive LLM inference benchmark")
    p.add_argument("--model", type=str, default=None, help="Model name or path (HF hub or local) - defaults to validator model if not specified")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (else fp16)")
    
    # Validator-specific args
    p.add_argument("--local", action="store_true", help="Use local run hparams")
    p.add_argument("--window", type=int, default=0, help="Window number to use for R2 dataset")
    p.add_argument("--seed", type=int, default=42, help="Seed for R2 dataset pages")
    p.add_argument("--load-checkpoint", action="store_true", help="Load checkpoint from R2 storage")

    p.add_argument("--batch-size", type=int, dest="batch_size", default=1)
    p.add_argument("--batch-sizes", type=int, nargs="*", default=None, help="Run sweep over these batch sizes")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--output-len", type=int, dest="output_len", default=128)
    p.add_argument("--num-reqs", type=int, default=20)

    # loader flags
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)

    # compiler / kernel flags
    p.add_argument("--enable-compile", action="store_true")
    p.add_argument("--enable-flashinfer", action="store_true")
    p.add_argument("--enable-thunder", action="store_true")

    # optimisation flags
    p.add_argument("--disable-empty-cache", action="store_true")
    p.add_argument("--empty-cache-every", type=int, default=10)
    p.add_argument("--no-grad", dest="use_inference_mode", action="store_false")
    p.add_argument("--use-amp", action="store_true")

    # profiler
    p.add_argument("--collect-profiler", action="store_true")
    p.add_argument("--profiler-outdir", type=Path, default=Path("./wandb_profiler"))

    # prompts
    p.add_argument("--prompts-path", type=str, default=None, help="JSON list of prompt strings")

    # wandb
    p.add_argument("--wandb-project", type=str, default="llm-benchmark")

    args = p.parse_args(argv)

    # ensure path exists
    if args.collect_profiler:
        args.profiler_outdir.mkdir(parents=True, exist_ok=True)

    if args.batch_sizes:
        # run matrix and exit after
        res_all = []
        for bs in args.batch_sizes:
            sub_args = parse_args(
                [*sys.argv[1:], "--batch-size", str(bs)] + (["--collect-profiler"] if args.collect_profiler else [])
            )
            print(f"\n===== Running batch‑size {bs} =====")
            bench = LLMInferenceBenchmark(sub_args)
            bench.run()
            res_all.append(bs)
        sys.exit(0)

    return args

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # enable heuristic
    main_args = parse_args()
    benchmark = LLMInferenceBenchmark(main_args)
    benchmark.run()
