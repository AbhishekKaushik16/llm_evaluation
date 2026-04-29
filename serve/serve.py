#!/usr/bin/env python3
"""Launch a vLLM inference server with OpenAI-compatible API.

Continuous batching and paged attention are enabled by default.
Exposes /v1/completions and /v1/chat/completions endpoints.

Two backends are supported:

  --backend metal  (default): use the vllm-metal plugin (Apple Silicon GPU).
                   Fast, but the 0.2.0 build does not implement logprobs;
                   loglikelihood-based evals will fail with HTTP 500.

  --backend cpu:   force upstream vLLM CPU. Sets VLLM_PLUGINS="" in the
                   child env so the metal platform plugin is skipped, and
                   tightens the memory budget to a CPU-safe default.
                   Slow, but logprobs / echo / prompt_logprobs work, so
                   loglikelihood evaluation produces real numbers.

Usage:
    python serve.py --model Qwen/Qwen2.5-3B-Instruct
    python serve.py --model Qwen/Qwen2.5-3B-Instruct --backend cpu
"""

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch vLLM OpenAI-compatible server")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Max context length. Defaults: 4096 for metal, 2048 for cpu "
            "(smaller default on cpu so the KV cache doesn't blow the RAM budget)."
        ),
    )
    parser.add_argument("--dtype", type=str, default="auto", help="Data type (auto, float16, bfloat16)")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "Fraction of device memory (Metal) or system RAM (CPU) to reserve "
            "for model weights + KV cache. Defaults: 0.90 for metal, 0.30 for cpu. "
            "On a 24 GB Mac, 0.30 = 7.2 GB — enough for a 3B bf16 model plus a "
            "modest KV cache. Increase if you have free RAM, decrease if vLLM "
            "complains about insufficient memory at startup."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        choices=["metal", "cpu"],
        default="metal",
        help=(
            "Inference backend. 'metal' uses the vllm-metal Apple Silicon "
            "plugin (fast, but logprobs are broken in 0.2.0). 'cpu' forces "
            "upstream vLLM CPU (slow, but logprobs work — required for "
            "loglikelihood-based evaluation)."
        ),
    )
    parser.add_argument(
        "--prefix-caching",
        choices=["on", "off"],
        default="on",
        help=(
            "Enable vLLM prefix caching (default: on). NB: disabling does NOT "
            "fix the vllm-metal logprobs/echo IndexError; that's a separate "
            "bug in the Metal port, not the upstream prefix-cache + logprobs "
            "interaction (vllm-project/vllm #5344). Exposed here for completeness."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = 0.30 if args.backend == "cpu" else 0.90
    if args.max_model_len is None:
        args.max_model_len = 2048 if args.backend == "cpu" else 4096

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--dtype", args.dtype,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--seed", str(args.seed),
    ]
    if args.prefix_caching == "on":
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")

    env = os.environ.copy()
    if args.backend == "cpu":
        # Skip the vllm-metal platform plugin so vLLM falls through to its
        # built-in CpuPlatform. Confirmed in vllm/plugins/__init__.py: empty
        # VLLM_PLUGINS produces an empty allow-list.
        env["VLLM_PLUGINS"] = ""
        # Compiling on CPU is rarely worth the warmup cost for short eval runs.
        cmd.append("--enforce-eager")
        print(
            "[serve] backend=cpu: VLLM_PLUGINS='' set; falling back to upstream "
            "vllm.platforms.cpu.CpuPlatform. Expect a slower init (~1-2 min "
            "for a 3B model) and slower throughput than metal."
        )

    print(f"Starting vLLM server (backend={args.backend}): {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Server exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
