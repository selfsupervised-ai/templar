"""Templar Llama Benchmarking Script

This script benchmarks inference throughput and latency of LlamaForCausalLM models used in the Templar architecture.
It supports evaluating both static checkpoints and the latest available model via the Bittensor/TPLR comms system.

Key Features:
    - Automatically fetches and loads the latest model checkpoint from R2.
    - Uses torch.profiler for optional performance tracing.
    - Detailed Benchmark Metrics: Computes throughput (req/s, tokens/s), latency (mean, median, p99), and GPU memory stats.
    - Flexible Execution Modes: Multiple benchmark profiles via `--exp`, supporting dataloader, compilation, and optimizer tuning.
    - Integration with wandb: All metrics are automatically logged for visualization.

Environment Requirements:
    - Registered Bittensor wallet
    - Access to TPLR-configured environment for fetching live checkpoints (env vars must be set) 
    - R2 Dataset access credentials
    - CUDA-enabled GPU (Ampere+ for BF16 support)

Usage Examples:
---------------
1. Run with latest model from Bittensor:
    python benchmark_llama_speed.py --exp full --use_latest --netuid 3

2. Run with specific checkpoint file:
    python benchmark_llama_speed.py --exp baseline --tplr_checkpoint /path/to/checkpoint.pt

3. Use compiler and FlashAttention:
    python benchmark_llama_speed.py --exp compile
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
                    nvmlInit, nvmlShutdown)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (LlamaForCausalLM, PreTrainedTokenizer,
                          default_data_collator, set_seed)

import wandb

try:
    import thunder  # type: ignore
except ImportError:
    thunder = None

# Bittensor & Templar
import bittensor as bt

import tplr  # provides load_hparams(), comms, logger, etc.


# ---------------------------------------------------------------------------
# Dataclasses & helpers
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkMetrics:
    """All the aspects of inference speed"""
    completed: int
    failures: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
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
    peak_gpu_memory_mib: float
    available_gpu_memory_mib: float
    gpu_utilization: float


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
        if self.enabled and hasattr(self.prof, "step"):
            self.prof.step()


# ---------------------------------------------------------------------------
# Bittensor helper – build minimal config from CLI
# ---------------------------------------------------------------------------

def build_bt_config(args: argparse.Namespace) -> bt.Config:
    """Create a `bt.Config` with subnet params + any Bittensor CLI flags."""
    p = argparse.ArgumentParser(add_help=False)
    bt.subtensor.add_args(p)
    bt.wallet.add_args(p)  # Add wallet args to ensure wallet config is initialized
    cli_args, _ = p.parse_known_args([])  # defaults only
    cfg = bt.config(p)
    
    # Explicitly initialize wallet config if it doesn't exist
    if not hasattr(cfg, 'wallet'):
        cfg.wallet = bt.config()
        
    cfg.wallet.name = args.wallet_name
    cfg.wallet.hotkey = args.wallet_hotkey
    cfg.netuid = args.netuid
    return cfg


async def _fetch_latest_checkpoint(cfg: bt.Config, hparams, netuid: int) -> Optional[Tuple[dict, tplr.comms.Comms]]:
    """Async wrapper replicating Evaluator.load_latest_model()."""
    try:
        wallet = bt.wallet(config=cfg)
        subtensor = bt.subtensor(config=cfg)
        metagraph = subtensor.metagraph(netuid=netuid)

        comms = tplr.comms.Comms(
            wallet=wallet,
            save_location="/tmp",
            key_prefix="model",
            config=cfg,
            netuid=netuid,
            metagraph=metagraph,
            hparams=hparams,
            uid=1,
        )
        result = await comms.get_latest_checkpoint(version=tplr.__version__)
        if not result:
            tplr.logger.warning("No checkpoint available via comms")
            return None
        checkpoint, _ = result
        return checkpoint, comms
    except Exception as e:
        tplr.logger.warning(f"Failed to fetch checkpoint via comms: {e}")
        return None


def load_llama_model(
    *,
    device: str,
    compile_mode: Optional[str] = None,
    tplr_checkpoint: Optional[str] = None,
    use_latest: bool = False,
    netuid: int = 3,
    bt_cfg: Optional[bt.Config] = None,
) -> Tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    """Load Templar Llama model.

    Priority: 1) `use_latest` via comms, 2) explicit `tplr_checkpoint`,
    3) random‑init (no weights).
    """
    hparams = tplr.load_hparams()
    model = LlamaForCausalLM(config=hparams.model_config)
    model.to(device=device, dtype=torch.bfloat16)

    try:
        if use_latest and tplr_checkpoint is None:
            cfg = bt_cfg or build_bt_config(argparse.Namespace(
                netuid=netuid,
                wallet_name=os.getenv("WALLET_NAME"),
                wallet_hotkey=os.getenv("WALLET_HOTKEY")
            ))
            tplr.logger.info("Fetching latest checkpoint via comms…")
            
            # Handle case where checkpoint fetching fails
            fetch_result = asyncio.run(_fetch_latest_checkpoint(cfg, hparams, netuid))
            if fetch_result is not None:
                checkpoint, _ = fetch_result
                state = {k: v.to(torch.bfloat16) for k, v in checkpoint["model_state_dict"].items()}
                model.load_state_dict(state)
            else:
                tplr.logger.warning("Failed to fetch checkpoint. Using random initialization.")
                # Continue with random initialization
        elif tplr_checkpoint:
            tplr.logger.info(f"Loading checkpoint from {tplr_checkpoint}")
            ckpt = torch.load(tplr_checkpoint, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()})
    except Exception as e:
        tplr.logger.warning(f"Error loading model: {e}")
        tplr.logger.info("Continuing with random initialization.")

    # optional compile
    if compile_mode == "thunder":
        if thunder is None:
            raise RuntimeError("thunder not installed – `pip install lightning‑thunder --pre`")
        model = thunder.jit(model, executors=["sdpa", "torchcompile_cat", "nvfuser", "torch"])
    elif compile_mode == "compile":
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    return model, hparams.tokenizer


# ---------------------------------------------------------------------------
# Benchmarking utils
# ---------------------------------------------------------------------------

def measure_throughput(
    model: torch.nn.Module,
    dataloader: DataLoader,
    steps: int,
    use_profiler: bool = False,
    profile_path: Optional[Path] = None,
    inference_mode: bool = False,
) -> BenchmarkMetrics:
    model.eval()
    profiler = Profiler(use_profiler, profile_path or Path("traces"))

    latencies: List[float] = []
    inputs: List[int] = []
    outputs: List[int] = []
    failures = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        avail_start, _ = torch.cuda.mem_get_info()
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        util_samples = []
    else:
        avail_start = 0

    t0_all = time.time()
    with profiler:
        for idx, batch in enumerate(tqdm(dataloader, total=steps, desc="benchmark")):
            if torch.cuda.is_available():
                util = nvmlDeviceGetUtilizationRates(handle)
                util_samples.append(util.gpu)
            if idx >= steps:
                break
            try:
                ts = time.time()
                batch = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
                if inference_mode:
                    with torch.inference_mode():
                        _ = model(**batch, use_cache=False)
                else:
                    with torch.no_grad():
                        _ = model(**batch, use_cache=False)
                te = time.time()

                in_tok = batch["input_ids"].numel()
                out_tok = in_tok  # forward‑only
                lat_ms = (te - ts) * 1_000

                latencies.append(lat_ms)
                inputs.append(in_tok)
                outputs.append(out_tok)
                profiler.step()
            except Exception as e:
                tplr.logger.error(f"inference error: {e}")
                failures += 1

    t1_all = time.time()
    total_s = t1_all - t0_all

    # stats helpers
    def stats(arr: List[float]) -> Tuple[float, float, float, float]:
        if not arr:
            return (0, 0, 0, 0)
        mean = statistics.mean(arr)
        median = statistics.median(arr)
        std = statistics.stdev(arr) if len(arr) > 1 else 0.0
        p99 = sorted(arr)[int(len(arr) * 0.99)]
        return (mean, median, std, p99)

    tt_mean, tt_med, tt_std, tt_p99 = stats(latencies)
    tpot_list = [l / o if o else 0 for l, o in zip(latencies, outputs)]
    tp_mean, tp_med, tp_std, tp_p99 = stats(tpot_list)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    avail_end = torch.cuda.mem_get_info()[0] / 1024**2 if torch.cuda.is_available() else 0
    avail_min = min(avail_start / 1024**2, avail_end) if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        nvmlShutdown()
        avg_util = sum(util_samples) / len(util_samples) if util_samples else 0.0
    else:
        avg_util = 0.0

    return BenchmarkMetrics(
        completed=len(latencies),
        failures=failures,
        total_input=sum(inputs),
        total_output=sum(outputs),
        request_throughput=len(latencies) / total_s if total_s else 0,
        input_throughput=sum(inputs) / total_s if total_s else 0,
        output_throughput=sum(outputs) / total_s if total_s else 0,
        mean_ttft_ms=tt_mean,
        median_ttft_ms=tt_med,
        std_ttft_ms=tt_std,
        p99_ttft_ms=tt_p99,
        mean_tpot_ms=tp_mean,
        median_tpot_ms=tp_med,
        std_tpot_ms=tp_std,
        p99_tpot_ms=tp_p99,
        mean_itl_ms=tt_mean,  # simplistic
        median_itl_ms=tt_med,
        std_itl_ms=tt_std,
        p99_itl_ms=tt_p99,
        max_input=max(inputs) if inputs else 0,
        max_output=max(outputs) if outputs else 0,
        max_total=(max(inputs) + max(outputs)) if inputs and outputs else 0,
        peak_gpu_memory_mib=peak_mem,
        available_gpu_memory_mib=avail_min,
        gpu_utilization=avg_util,
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_dataset(name: str, seq_len: int, num_samples: int = 4096):
    return load_dataset(name, split=f"train[:{num_samples}]")


def tokenize(example, tokenizer, seq_len: int):
    return tokenizer(example["text"], truncation=True, max_length=seq_len, padding="max_length")


def build_dataloader(ds, tokenizer, seq_len, batch_size, pin_mem, persistent, collate_cuda):
    def _collate(b):
        b = default_data_collator(b)
        if collate_cuda:
            return {k: v.to("cuda", non_blocking=True) for k, v in b.items()}
        return b

    tokenised = ds.map(partial(tokenize, tokenizer=tokenizer, seq_len=seq_len), batched=False)
    return DataLoader(
        tokenised,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        collate_fn=_collate,
    )


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
Experiment = Callable[[argparse.Namespace], None]
registry: Dict[str, Experiment] = {}

def register(name):
    def wrap(fn):
        registry[name] = fn
        return fn
    return wrap


# Base experiment template

def run_experiment(args, compile_mode=None, pin_mem=False, persistent=False, collate_cuda=False, profile=False, inference_mode=False):
    model, tokenizer = load_llama_model(
        device=args.device,
        compile_mode=compile_mode,
        tplr_checkpoint=args.tplr_checkpoint,
        use_latest=args.use_latest,
        netuid=args.netuid,
    )
    ds = get_dataset(args.dataset, args.seq_len)
    dl = build_dataloader(ds, tokenizer, args.seq_len, args.batch_size, pin_mem, persistent, collate_cuda)

    # Warm-up: run a few batches to trigger compilation, caching, etc.
    model.eval()
    warmup_batches = 3
    for idx, batch in enumerate(dl):
        if idx >= warmup_batches:
            break
        batch = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
        with torch.inference_mode():
            _ = model(**batch, use_cache=False)

    profile_path = Path("traces") / f"{args.exp}_{time.time_ns()}" if profile else None
    metrics = measure_throughput(model, dl, args.steps, use_profiler=profile, profile_path=profile_path, inference_mode=inference_mode)

    # Log to wandb
    wandb.log(metrics.__dict__)


@register("baseline")
def exp_baseline(args):
    run_experiment(args, profile=True)


@register("simple")
def exp_simple(args):
    torch.backends.cudnn.benchmark = True
    run_experiment(args, pin_mem=True, persistent=True, inference_mode=True)


@register("compile")
def exp_compile(args):
    torch.backends.cuda.enable_flash_sdp(True)
    run_experiment(args, compile_mode="compile", pin_mem=True, persistent=True, inference_mode=True)

@register("thunder")
def exp_thunder(args):
    torch.backends.cuda.enable_flash_sdp(True)
    run_experiment(args, compile_mode="thunder", pin_mem=True, persistent=True, inference_mode=True)

@register("dataloader")
def exp_dataloader(args):
    run_experiment(args, pin_mem=True, persistent=True, collate_cuda=False, inference_mode=True)


@register("full")
def exp_full(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    mode = "compile"
    run_experiment(args, compile_mode=mode, pin_mem=True, persistent=True, collate_cuda=False, inference_mode=True)


# ---------------------------------------------------------------------------
# CLI & entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Templar Llama speed ablation with live checkpoints")
    p.add_argument("--exp", choices=list(registry), help="experiment name")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--steps", type=int, default=40, help="forward passes to measure")
    p.add_argument("--dataset", default="abcorrea/wikitest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tplr_checkpoint", type=str, default=None)
    p.add_argument("--use_latest", default=True)
    p.add_argument("--netuid", type=int, default=3)
    p.add_argument("--wallet_name", default=os.getenv("WALLET_NAME"))
    p.add_argument("--wallet_hotkey", default=os.getenv("WALLET_HOTKEY"))
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    set_seed(args.seed)
    Path("traces").mkdir(exist_ok=True)
    wandb.init(project="templar_llama_speed", name=f"{args.exp}_{time.time_ns()}", config=vars(args))

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    registry[args.exp](args)
    wandb.finish()


if __name__ == "__main__":
    main()
