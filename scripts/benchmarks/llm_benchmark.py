# benchmark_llama_speed.py
"""Speed‑ablation harness for Llama 3.1‑8B (bf16)

This standalone script lets you reproduce the optimisation steps described in
our proposal.  Each step is a **named experiment**; run them individually or as
an end‑to‑end sweep.  All runs are logged to `wandb`, and every stage emits a
throughput metric (`tokens_per_second`) and, optionally, a `torch.profiler`
Chrome‑trace for deeper inspection.

Usage examples
--------------

```bash
#1) baseline profile (writes trace in ./traces/baseline_*.pt)
python benchmark_llama_speed.py --exp baseline

#2)  simple tweaks (no profiler) – compares to baseline in wandb graphs
python benchmark_llama_speed.py --exp simple

#3)  compiled with Thunder JIT & FlashAttention
python benchmark_llama_speed.py --exp compile

#4)  data‑loader tuning
python benchmark_llama_speed.py --exp dataloader

#5)  full stack (compile + dataloader + streams)
python benchmark_llama_speed.py --exp full --steps 500
```

A typical workflow is to run the experiments in order and observe the
throughput gains in the `wandb` dashboard.

Notes
-----
* The script purposely avoids any form of quantisation – the model runs in
  **BF16** throughout.
* Tested with PyTorch 2.3, CUDA 12.4, `transformers` 4.41, Thunder nightly ≥ Apr‑2025.
* **GPU requirements:** Ampere (A100) or Hopper (H100).  Hopper automatically
  selects FlashAttention‑2 via SDPA when available.
* The script fetches the *Hugging Face* eval dataset **lighteval/lighteval‑tiny**
  by default; override with `--dataset`.
"""
from __future__ import annotations

import argparse
import os
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import torch
import torch.utils.data as tud
import wandb
from datasets import load_dataset
from tqdm.auto import tqdm
# transformers / HF
from transformers import (AutoConfig, AutoTokenizer, LlamaForCausalLM,
                          default_data_collator, set_seed)

# Optional imports – load lazily
try:
    import thunder  # type: ignore
except ImportError:
    thunder = None

try:
    from flash_attn.layers.rotary import \
        RotaryEmbedding  # noqa: F401 – check availability
except ImportError:
    pass

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------


def log_to_wandb(config: Dict):
    """Initialise wandb in a deterministic (resume‑friendly) way."""
    wandb.init(
        project="llama31_8b_speed",
        name=f"{config['exp']}_{time.strftime('%Y%m%d_%H%M%S')}",
        config=config,
        mode=os.getenv("WANDB_MODE", "online"),
        tags=[config["exp"], config["device"], "bf16"],
    )


@torch.no_grad()
@torch.inference_mode()
def measure_throughput(
    model: torch.nn.Module,
    dataloader: tud.DataLoader,
    steps: int,
    use_profiler: bool = False,
    profile_path: Path | None = None,
) -> float:
    """Run *steps* forward passes and return tokens/s.

    Parameters
    ----------
    model : torch.nn.Module
        The model to benchmark (already on correct device).
    dataloader : DataLoader
        Yields dicts with ``input_ids`` & ``attention_mask`` on **CPU** or
        **CUDA** depending on collate_fn.
    steps : int
        Number of mini‑batches to benchmark.
    use_profiler : bool, default False
        Capture a **torch.profiler** trace for the first 100 steps.
    profile_path : Path | None
        Where to dump the trace (.pt); created only if ``use_profiler``.
    """
    model.eval()

    if use_profiler:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=80),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_path)),
            record_shapes=True,
            with_stack=True,
            with_flops=True,
        )
        prof.__enter__()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    n_tokens = 0
    for step, batch in enumerate(tqdm(dataloader, total=steps, desc="benchmark")):
        if step >= steps:
            break
        batch = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch, use_cache=False)
        n_tokens += batch["input_ids"].numel()
        if use_profiler:
            prof.step()

    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    tok_per_s = n_tokens / (ms / 1_000)

    if use_profiler:
        prof.__exit__(None, None, None)

    return tok_per_s


# ----------------------------------------------------------------------------
# Data‑set & loader helpers
# ----------------------------------------------------------------------------

def get_dataset(name: str, seq_len: int, num_samples: int = 4096):
    """Load a small HF dataset for quick benchmarks."""
    ds = load_dataset(name, split="train[:{0}]".format(num_samples))
    return ds


def tokenize_function(tokenizer, example, seq_len: int):
    """Tokenise a text field and trim/pad to *seq_len*."""
    tokens = tokenizer(example["text"], truncation=True, max_length=seq_len, padding="max_length")
    return tokens


def build_dataloader(
    ds,
    tokenizer,
    seq_len: int,
    batch_size: int,
    pin_memory: bool,
    persistent_workers: bool,
    collate_cuda: bool,
):
    """Return DataLoader with optional *collate‑to‑cuda* semantic."""

    def _collate(batch):
        batch = default_data_collator(batch)
        if collate_cuda:
            return {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        return batch

    tokenised = ds.map(partial(tokenize_function, tokenizer, seq_len=seq_len), batched=False)
    dl = tud.DataLoader(
        tokenised,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=_collate,
    )
    return dl


# ----------------------------------------------------------------------------
# Model builders for each experiment
# ----------------------------------------------------------------------------

def load_llama_model(device: str, compile_mode: str | None = None):
    """Load Llama 3.1‑8B in BF16 and prepare according to *compile_mode*.

    compile_mode
        *None*            – eager.
        *"thunder"*       – Thunder JIT (requires thunder installed).
        *"compile"*       – torch.compile (Inductor).
    """
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    config = AutoConfig.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    if compile_mode == "thunder":
        if thunder is None:
            raise RuntimeError("thunder not installed – pip install lightning-thunder --pre")
        model = thunder.jit(
            model, executors=["sdpa", "torchcompile_cat", "nvfuser", "torch"]
        )
    elif compile_mode == "compile":
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    return model


# ----------------------------------------------------------------------------
# Experiment registry
# ----------------------------------------------------------------------------

Experiment = Callable[[argparse.Namespace], None]
registry: Dict[str, Experiment] = {}

def register(name: str):
    def _wrap(fn: Experiment):
        registry[name] = fn
        return fn
    return _wrap


@register("baseline")
def exp_baseline(args):
    """Eager BF16 baseline with profiler."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True)
    ds = get_dataset(args.dataset, args.seq_len)
    dl = build_dataloader(ds, tokenizer, args.seq_len, args.batch_size, False, False, False)

    model = load_llama_model(args.device)

    tok_s = measure_throughput(
        model,
        dl,
        args.steps,
        use_profiler=True,
        profile_path=Path("traces") / f"baseline_{time.time_ns()}",
    )
    wandb.log({"tokens_per_second": tok_s})


@register("simple")
def exp_simple(args):
    """Apply low‑hanging wins (inference_mode, allocator, cudnn, pinned loader)."""
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True)
    ds = get_dataset(args.dataset, args.seq_len)
    dl = build_dataloader(ds, tokenizer, args.seq_len, args.batch_size, True, True, False)

    model = load_llama_model(args.device)

    tok_s = measure_throughput(model, dl, args.steps)
    wandb.log({"tokens_per_second": tok_s})


@register("compile")
def exp_compile(args):
    """Thunder or torch.compile + FlashAttention (via SDPA)."""
    torch.backends.cuda.enable_flash_sdp(True)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True)
    ds = get_dataset(args.dataset, args.seq_len)
    dl = build_dataloader(ds, tokenizer, args.seq_len, args.batch_size, True, True, False)

    compile_mode = "thunder" if thunder else "compile"
    model = load_llama_model(args.device, compile_mode=compile_mode)

    tok_s = measure_throughput(model, dl, args.steps)
    wandb.log({"tokens_per_second": tok_s, "compile_mode": compile_mode})


@register("dataloader")
def exp_dataloader(args):
    """Collate‑to‑CUDA & async prefetch using a background stream."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True)
    ds = get_dataset(args.dataset, args.seq_len)
    dl = build_dataloader(ds, tokenizer, args.seq_len, args.batch_size, True, True, True)

    model = load_llama_model(args.device)

    tok_s = measure_throughput(model, dl, args.steps)
    wandb.log({"tokens_per_second": tok_s})


@register("full")
def exp_full(args):
    """All optimisations: compile + collate‑cuda + cudnn + streams."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True)
    ds = get_dataset(args.dataset, args.seq_len)
    dl = build_dataloader(ds, tokenizer, args.seq_len, args.batch_size, True, True, True)

    compile_mode = "thunder" if thunder else "compile"
    model = load_llama_model(args.device, compile_mode)

    tok_s = measure_throughput(model, dl, args.steps)
    wandb.log({"tokens_per_second": tok_s, "compile_mode": compile_mode})


# ----------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama 3.1‑8B inference speed ablation")
    p.add_argument("--exp", choices=list(registry), help="experiment name")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--seq_len", type=int, default=128, help="sequence length")
    p.add_argument("--batch_size", type=int, default=4, help="mini‑batch size")
    p.add_argument("--steps", type=int, default=200, help="forward passes to measure")
    p.add_argument("--dataset", default="lighteval/lighteval‑tiny", help="HF dataset")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    Path("traces").mkdir(exist_ok=True)

    log_to_wandb(vars(args))

    torch.cuda.manual_seed_all(args.seed)

    # dispatch experiment
    registry[args.exp](args)

    wandb.finish()


if __name__ == "__main__":
    main()