#!/usr/bin/env python3
"""Load generator for vLLM performance benchmarking.

Sends concurrent requests (short vs long prompts) and collects:
  - Time-to-first-token (TTFT)
  - Tokens per second (throughput)
  - P50 / P95 / P99 latency
  - GPU utilization (Apple Silicon, best-effort)

Usage:
    python load_test.py --base-url http://localhost:8000 --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import asyncio
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import numpy as np
from tqdm import tqdm


SHORT_PROMPTS = [
    "What is 2 + 2?",
    "Name a color.",
    "Say hello.",
    "What is the capital of France?",
    "Define gravity in one sentence.",
    "What year did World War II end?",
    "Name a planet in our solar system.",
    "What is H2O?",
]

LONG_PROMPTS = [
    "Write a detailed essay about the history of artificial intelligence, covering its origins in the 1950s, the AI winters, the resurgence with machine learning, and the current state of large language models. Discuss the key milestones, important researchers, and the societal implications of each era. Be thorough and provide specific examples.",
    "Explain the complete process of how a bill becomes a law in the United States, starting from when a member of Congress drafts the initial proposal, through committee review, floor debate, reconciliation between the House and Senate versions, and finally presidential action. Include details about filibusters, vetoes, and override procedures.",
    "Describe the water cycle in exhaustive detail, including evaporation from oceans and lakes, transpiration from plants, condensation into clouds, various forms of precipitation, surface runoff, groundwater infiltration, and how human activities like deforestation and urbanization affect each stage of this cycle.",
    "Provide a comprehensive overview of the human cardiovascular system, including the structure and function of the heart's four chambers, the pulmonary and systemic circulation loops, the role of arteries, veins, and capillaries, blood composition, and common cardiovascular diseases along with their risk factors and prevention strategies.",
]


@dataclass
class RequestMetrics:
    concurrency: int
    prompt_type: str
    ttft_ms: float
    tpot: float  # tokens per second
    total_latency_ms: float
    tokens_generated: int
    stop_setting: str


async def make_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    stop: list[str] | None,
) -> dict:
    """Send a streaming request and collect timing metrics."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": True,
    }
    if stop:
        payload["stop"] = stop

    t_start = time.perf_counter()
    ttft = None
    token_count = 0

    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded or not decoded.startswith("data: "):
                continue
            data_str = decoded[len("data: "):]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                token = chunk["choices"][0].get("text", "")
                if token:
                    if ttft is None:
                        ttft = (time.perf_counter() - t_start) * 1000
                    token_count += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    total_ms = (time.perf_counter() - t_start) * 1000

    if ttft is None:
        ttft = total_ms

    generation_time_s = (total_ms - ttft) / 1000 if total_ms > ttft else total_ms / 1000
    tpot = token_count / generation_time_s if generation_time_s > 0 else 0.0

    return {
        "ttft_ms": ttft,
        "tpot": tpot,
        "total_latency_ms": total_ms,
        "tokens_generated": token_count,
    }


async def run_batch(
    url: str,
    model: str,
    prompts: list[str],
    prompt_type: str,
    concurrency: int,
    max_tokens: int,
    stop: list[str] | None,
    stop_label: str,
) -> list[RequestMetrics]:
    """Run a batch of concurrent requests."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Cycle prompts to fill the concurrency level
        batch_prompts = [prompts[i % len(prompts)] for i in range(concurrency)]

        tasks = [
            make_request(session, url, model, p, max_tokens, stop)
            for p in batch_prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    metrics = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  [ERROR] {r}")
            continue
        metrics.append(RequestMetrics(
            concurrency=concurrency,
            prompt_type=prompt_type,
            ttft_ms=r["ttft_ms"],
            tpot=r["tpot"],
            total_latency_ms=r["total_latency_ms"],
            tokens_generated=r["tokens_generated"],
            stop_setting=stop_label,
        ))
    return metrics


def compute_percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0}
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
    }


def try_gpu_utilization() -> float | None:
    """Attempt to read GPU utilization on Apple Silicon via powermetrics."""
    try:
        result = subprocess.run(
            ["sudo", "powermetrics", "--samplers", "gpu_power", "-n", "1", "-i", "1000"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "GPU Active" in line:
                parts = line.split()
                for p in parts:
                    if p.endswith("%"):
                        return float(p.strip("%"))
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        pass
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM load testing")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=str, default="perf/metrics.csv")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=3, help="Repeat each config N times")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    completions_url = f"{args.base_url}/v1/completions"
    all_metrics: list[RequestMetrics] = []

    stop_configs = [
        (None, "none"),
        (["\n\n"], "double_newline"),
        (["\n"], "single_newline"),
    ]

    prompt_configs = [
        (SHORT_PROMPTS, "short"),
        (LONG_PROMPTS, "long"),
    ]

    total_runs = len(args.concurrency_levels) * len(prompt_configs) * len(stop_configs) * args.repeats
    pbar = tqdm(total=total_runs, desc="Load testing")

    for concurrency in args.concurrency_levels:
        for prompts, prompt_type in prompt_configs:
            for stop, stop_label in stop_configs:
                for _ in range(args.repeats):
                    try:
                        batch_metrics = await run_batch(
                            completions_url, args.model, prompts, prompt_type,
                            concurrency, args.max_tokens, stop, stop_label,
                        )
                        all_metrics.extend(batch_metrics)
                    except Exception as e:
                        print(f"  [BATCH ERROR] concurrency={concurrency} "
                              f"type={prompt_type} stop={stop_label}: {e}")
                    pbar.update(1)

    pbar.close()

    # Write raw metrics to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "concurrency", "prompt_type", "ttft_ms", "tpot",
            "total_latency_ms", "tokens_generated", "stop_setting",
        ])
        for m in all_metrics:
            writer.writerow([
                m.concurrency, m.prompt_type, f"{m.ttft_ms:.2f}",
                f"{m.tpot:.2f}", f"{m.total_latency_ms:.2f}",
                m.tokens_generated, m.stop_setting,
            ])

    print(f"\nWrote {len(all_metrics)} records to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Concurrency':<12} {'Type':<8} {'Stop':<16} {'TTFT P50':>10} {'TTFT P95':>10} "
          f"{'Lat P50':>10} {'Lat P95':>10} {'Tok/s':>8}")
    print("-" * 86)

    for concurrency in args.concurrency_levels:
        for _, prompt_type in prompt_configs:
            for _, stop_label in stop_configs:
                subset = [m for m in all_metrics
                          if m.concurrency == concurrency
                          and m.prompt_type == prompt_type
                          and m.stop_setting == stop_label]
                if not subset:
                    continue
                ttft_stats = compute_percentiles([m.ttft_ms for m in subset])
                lat_stats = compute_percentiles([m.total_latency_ms for m in subset])
                tpot_stats = compute_percentiles([m.tpot for m in subset])
                print(f"{concurrency:<12} {prompt_type:<8} {stop_label:<16} "
                      f"{ttft_stats['p50']:>10.1f} {ttft_stats['p95']:>10.1f} "
                      f"{lat_stats['p50']:>10.1f} {lat_stats['p95']:>10.1f} "
                      f"{tpot_stats['mean']:>8.1f}")

    gpu_util = try_gpu_utilization()
    if gpu_util is not None:
        print(f"\nGPU Active: {gpu_util:.1f}%")
    else:
        print("\nGPU utilization: not available (requires sudo powermetrics on macOS)")


if __name__ == "__main__":
    asyncio.run(main())
