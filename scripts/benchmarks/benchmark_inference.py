#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for profiling LLM inference performance.
This script loads a LlamaForCausalLM model with the same configuration as in the validator,
but uses pre-trained weights from Huggingface. It profiles inference performance and logs
results to Weights & Biases.
"""

import argparse
import os
import time

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.profiler
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, LlamaForCausalLM

import tplr
from tplr.compress import TransformDCT


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference performance")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Huggingface model name",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run inference on"
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated list of batch sizes to test",
    )
    parser.add_argument(
        "--sequence_lengths",
        type=str,
        default="128,512,1024,2048",
        help="Comma-separated list of sequence lengths to test",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of iterations for each configuration",
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=3, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable detailed profiling"
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Compilation mode for torch.compile",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llm-inference-benchmark",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="W&B entity name"
    )
    parser.add_argument(
        "--inference_mode",
        choices=["no_grad", "inference_mode"],
        default="inference_mode",
        help="Inference context manager to use",
    )
    parser.add_argument(
        "--empty_cache", action="store_true", help="Empty CUDA cache between iterations"
    )

    return parser.parse_args()


class LLMBenchmark(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # Load hparams from tplr to match validator configuration
        self.tplr_hparams = tplr.load_hparams()

        # Load model and tokenizer
        print(f"Loading model {args.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # Configure model with same config as validator
        if hasattr(self.tplr_hparams, "model_config"):
            print("Using model config from tplr hparams")
            self.model = LlamaForCausalLM(self.tplr_hparams.model_config)
            # Load pre-trained weights
            self.model.load_state_dict(
                LlamaForCausalLM.from_pretrained(args.model_name).state_dict()
            )
        else:
            print("Using default model from Huggingface")
            self.model = LlamaForCausalLM.from_pretrained(args.model_name)

        # Move model to device
        self.model = self.model.to(args.device)

        # Compile model if requested
        if args.compile:
            print(f"Compiling model with mode: {args.compile_mode}")
            self.model = torch.compile(self.model, mode=args.compile_mode)

        # Initialize DCT transformer for comparison with validator
        self.transformer = TransformDCT(
            self.model, target_chunk=getattr(self.tplr_hparams, "target_chunk", 32)
        )

        # Set up benchmark metrics
        self.metrics = {
            "batch_size": [],
            "sequence_length": [],
            "iteration": [],
            "latency": [],
            "tokens_per_second": [],
            "memory_used": [],
            "compilation_mode": [],
            "inference_mode": [],
        }

        # Configure CUDA benchmarking
        torch.backends.cudnn.benchmark = True

    def generate_sample_batch(self, batch_size, sequence_length):
        """Generate a sample batch of input IDs."""
        # Create a prompt template
        prompt = "Once upon a time, there was a"

        # Tokenize the prompt
        encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = encoded.input_ids

        # Pad or truncate to desired sequence length
        if prompt_ids.shape[1] < sequence_length:
            # Pad with attention mask
            padding_length = sequence_length - prompt_ids.shape[1]
            pad_token_id = self.tokenizer.pad_token_id or 0

            # Create padded input_ids
            input_ids = torch.cat(
                [
                    prompt_ids,
                    torch.full((1, padding_length), pad_token_id, dtype=torch.long),
                ],
                dim=1,
            )

            # Create attention mask
            attention_mask = torch.cat(
                [torch.ones(1, prompt_ids.shape[1]), torch.zeros(1, padding_length)],
                dim=1,
            )
        else:
            # Truncate
            input_ids = prompt_ids[:, :sequence_length]
            attention_mask = torch.ones(1, sequence_length)

        # Repeat for batch size
        input_ids = input_ids.repeat(batch_size, 1)
        attention_mask = attention_mask.repeat(batch_size, 1)

        return {
            "input_ids": input_ids.to(self.args.device),
            "attention_mask": attention_mask.to(self.args.device),
        }

    def run_inference(self, batch, profile=False):
        """Run inference on a batch with optional profiling."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Create labels for loss calculation (same as input_ids)
        labels = input_ids.clone()
        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

        # Choose inference context based on args
        context_manager = (
            torch.inference_mode
            if self.args.inference_mode == "inference_mode"
            else torch.no_grad
        )

        # Run inference with profiling if requested
        if profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            ) as prof:
                with context_manager():
                    self.model.eval()
                    # Warmup
                    for _ in range(self.args.warmup_iterations):
                        with autocast(
                            device_type=self.args.device, dtype=torch.bfloat16
                        ):
                            _ = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                        prof.step()

                    # Actual profiling
                    latencies = []
                    for _ in range(self.args.num_iterations):
                        start_time = time.time()
                        with autocast(
                            device_type=self.args.device, dtype=torch.bfloat16
                        ):
                            _ = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                        torch.cuda.synchronize()
                        latencies.append(time.time() - start_time)
                        prof.step()

            # Export trace
            os.makedirs(self.args.output_dir, exist_ok=True)
            trace_path = os.path.join(
                self.args.output_dir,
                f"trace_bs{input_ids.shape[0]}_seq{input_ids.shape[1]}.json",
            )
            prof.export_chrome_trace(trace_path)
            print(f"Exported profiler trace to {trace_path}")

            # Print summary
            print("\nProfiler Summary (top 20 ops by CUDA time):")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

            return np.mean(latencies), prof
        else:
            # Run without detailed profiling
            with context_manager():
                self.model.eval()
                # Warmup
                for _ in range(self.args.warmup_iterations):
                    with autocast(device_type=self.args.device, dtype=torch.bfloat16):
                        _ = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )

                # Actual benchmark
                latencies = []
                for _ in range(self.args.num_iterations):
                    # Clear cache if requested
                    if self.args.empty_cache:
                        torch.cuda.empty_cache()

                    start_time = time.time()
                    with autocast(device_type=self.args.device, dtype=torch.bfloat16):
                        _ = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                    torch.cuda.synchronize()
                    latencies.append(time.time() - start_time)

            return np.mean(latencies), None

    def benchmark_configuration(self, batch_size, sequence_length, profile=False):
        """Benchmark a specific configuration of batch size and sequence length."""
        print(
            f"\nBenchmarking batch_size={batch_size}, sequence_length={sequence_length}"
        )

        # Generate sample batch
        batch = self.generate_sample_batch(batch_size, sequence_length)

        # Record memory before inference
        torch.cuda.reset_peak_memory_stats()

        # Run inference
        latency, prof = self.run_inference(batch, profile=profile)

        # Calculate metrics
        tokens_per_second = (batch_size * sequence_length) / latency
        memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB

        print(f"Latency: {latency * 1000:.2f} ms")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        print(f"Memory used: {memory_used:.2f} MB")

        # Record metrics
        self.metrics["batch_size"].append(batch_size)
        self.metrics["sequence_length"].append(sequence_length)
        self.metrics["iteration"].append(len(self.metrics["latency"]))
        self.metrics["latency"].append(latency)
        self.metrics["tokens_per_second"].append(tokens_per_second)
        self.metrics["memory_used"].append(memory_used)
        self.metrics["compilation_mode"].append(
            self.args.compile_mode if self.args.compile else "none"
        )
        self.metrics["inference_mode"].append(self.args.inference_mode)

        # Log to W&B
        if self.logger:
            self.logger.experiment.log(
                {
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "latency": latency,
                    "tokens_per_second": tokens_per_second,
                    "memory_used": memory_used,
                    "compilation_mode": self.args.compile_mode
                    if self.args.compile
                    else "none",
                    "inference_mode": self.args.inference_mode,
                }
            )

        return latency, tokens_per_second, memory_used, prof

    def run_all_benchmarks(self):
        """Run benchmarks for all configurations."""
        batch_sizes = [int(bs) for bs in self.args.batch_sizes.split(",")]
        sequence_lengths = [int(sl) for sl in self.args.sequence_lengths.split(",")]

        results = []

        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                # Run with profiling for the first iteration if requested
                profile_this_config = self.args.profile and len(results) == 0
                latency, tokens_per_second, memory_used, _ = (
                    self.benchmark_configuration(
                        batch_size, sequence_length, profile=profile_this_config
                    )
                )

                results.append(
                    {
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "latency": latency,
                        "tokens_per_second": tokens_per_second,
                        "memory_used": memory_used,
                    }
                )

        # Create DataFrame from results
        df = pd.DataFrame(results)

        # Save results
        os.makedirs(self.args.output_dir, exist_ok=True)
        df.to_csv(
            os.path.join(self.args.output_dir, "benchmark_results.csv"), index=False
        )

        # Create plots
        self.create_plots(df)

        return df

    def create_plots(self, df):
        """Create and save plots from benchmark results."""
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Plot 1: Tokens per second vs Batch Size for different sequence lengths
        plt.figure(figsize=(10, 6))
        for seq_len in df["sequence_length"].unique():
            subset = df[df["sequence_length"] == seq_len]
            plt.plot(
                subset["batch_size"],
                subset["tokens_per_second"],
                marker="o",
                label=f"Seq Len: {seq_len}",
            )

        plt.xlabel("Batch Size")
        plt.ylabel("Tokens per Second")
        plt.title("Inference Performance: Tokens per Second vs Batch Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self.args.output_dir, "tokens_per_second_vs_batch_size.png")
        )

        # Plot 2: Memory Usage vs Batch Size for different sequence lengths
        plt.figure(figsize=(10, 6))
        for seq_len in df["sequence_length"].unique():
            subset = df[df["sequence_length"] == seq_len]
            plt.plot(
                subset["batch_size"],
                subset["memory_used"],
                marker="o",
                label=f"Seq Len: {seq_len}",
            )

        plt.xlabel("Batch Size")
        plt.ylabel("Memory Used (MB)")
        plt.title("Memory Usage vs Batch Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.args.output_dir, "memory_vs_batch_size.png"))

        # Plot 3: Latency vs Batch Size for different sequence lengths
        plt.figure(figsize=(10, 6))
        for seq_len in df["sequence_length"].unique():
            subset = df[df["sequence_length"] == seq_len]
            plt.plot(
                subset["batch_size"],
                subset["latency"] * 1000,
                marker="o",
                label=f"Seq Len: {seq_len}",
            )

        plt.xlabel("Batch Size")
        plt.ylabel("Latency (ms)")
        plt.title("Inference Latency vs Batch Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.args.output_dir, "latency_vs_batch_size.png"))

        # Plot 4: Heatmap of tokens per second
        pivot_df = df.pivot_table(
            values="tokens_per_second", index="sequence_length", columns="batch_size"
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="viridis")
        plt.title("Tokens per Second Heatmap")
        plt.savefig(os.path.join(self.args.output_dir, "tokens_per_second_heatmap.png"))

        # Log plots to W&B
        if self.logger:
            self.logger.experiment.log(
                {
                    "tokens_per_second_vs_batch_size": wandb.Image(
                        os.path.join(
                            self.args.output_dir, "tokens_per_second_vs_batch_size.png"
                        )
                    ),
                    "memory_vs_batch_size": wandb.Image(
                        os.path.join(self.args.output_dir, "memory_vs_batch_size.png")
                    ),
                    "latency_vs_batch_size": wandb.Image(
                        os.path.join(self.args.output_dir, "latency_vs_batch_size.png")
                    ),
                    "tokens_per_second_heatmap": wandb.Image(
                        os.path.join(
                            self.args.output_dir, "tokens_per_second_heatmap.png"
                        )
                    ),
                }
            )


def main():
    # Parse arguments
    args = parse_args()

    # Enable debug logging if requested
    if args.debug:
        tplr.debug()

    # Set up W&B logger
    try:
        import wandb

        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"llm-inference-benchmark-{args.model_name.split('/')[-1]}",
            config=vars(args),
        )
    except ImportError:
        print("W&B not available, continuing without logging")
        wandb_logger = None

    # Create benchmark module
    benchmark = LLMBenchmark(args)

    # Set logger
    benchmark.logger = wandb_logger

    # Run benchmarks
    results_df = benchmark.run_all_benchmarks()

    # Print summary
    print("\nBenchmark Summary:")
    print(results_df)

    # Close W&B
    if wandb_logger:
        wandb.finish()


if __name__ == "__main__":
    main()
