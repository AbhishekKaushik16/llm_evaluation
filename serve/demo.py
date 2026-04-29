#!/usr/bin/env python3
"""Demo script: runs sample prompt generations and a concurrency test.

Usage:
    python demo.py --base-url http://localhost:8000 --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import asyncio
import time
import sys

sys.path.insert(0, ".")
from serve.client import VLLMClient


SAMPLE_PROMPTS = [
    "Explain quantum entanglement in one paragraph:",
    "Write a haiku about machine learning:",
    "What are the three laws of thermodynamics? Be concise.",
    "Translate to French: The weather is beautiful today.",
    "Write a Python function that checks if a number is prime:",
]

CONCURRENCY_PROMPTS = [
    "What is the capital of France?",
    "Summarize the theory of relativity in two sentences.",
    "Name five programming languages and their main use cases.",
    "What is photosynthesis?",
    "Explain the difference between TCP and UDP.",
]


async def demo_streaming(client: VLLMClient) -> None:
    print("=" * 60)
    print("STREAMING GENERATION DEMO")
    print("=" * 60)

    prompt = "Explain why the sky is blue in simple terms:"
    print(f"\nPrompt: {prompt}\n")
    print("Response: ", end="", flush=True)

    t0 = time.perf_counter()
    ttft = None
    token_count = 0
    async for token in client.generate_stream(
        prompt, max_tokens=150, temperature=0.7, top_p=0.95
    ):
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        print(token, end="", flush=True)
        token_count += 1
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n\n[TTFT: {ttft:.0f}ms | Tokens: {token_count} | "
          f"Total: {elapsed:.0f}ms | Speed: {token_count / (elapsed / 1000):.1f} tok/s]")


async def demo_batch(client: VLLMClient) -> None:
    print("\n" + "=" * 60)
    print("SEQUENTIAL GENERATION DEMO")
    print("=" * 60)

    for i, prompt in enumerate(SAMPLE_PROMPTS, 1):
        print(f"\n--- Prompt {i} ---")
        print(f"Prompt: {prompt}")
        result = await client.generate(
            prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.95,
            stop=["\n\n"],
        )
        print(f"Response: {result.text.strip()}")
        print(f"[Tokens: {result.tokens_generated} | "
              f"Time: {result.elapsed_seconds:.2f}s | "
              f"Speed: {result.tokens_generated / result.elapsed_seconds:.1f} tok/s]")


async def demo_concurrent(client: VLLMClient) -> None:
    print("\n" + "=" * 60)
    print("CONCURRENT GENERATION DEMO (5 parallel requests)")
    print("=" * 60)

    t0 = time.perf_counter()
    results = await client.generate_concurrent(
        CONCURRENCY_PROMPTS,
        max_tokens=80,
        temperature=0.7,
        top_p=0.95,
    )
    total_time = time.perf_counter() - t0

    for i, (prompt, result) in enumerate(zip(CONCURRENCY_PROMPTS, results), 1):
        print(f"\n--- Request {i} ---")
        print(f"Prompt: {prompt}")
        print(f"Response: {result.text.strip()[:120]}...")
        print(f"[Tokens: {result.tokens_generated} | Time: {result.elapsed_seconds:.2f}s]")

    total_tokens = sum(r.tokens_generated for r in results)
    avg_time = sum(r.elapsed_seconds for r in results) / len(results)
    print(f"\n{'=' * 60}")
    print(f"CONCURRENCY SUMMARY")
    print(f"  Total wall-clock time: {total_time:.2f}s")
    print(f"  Average per-request time: {avg_time:.2f}s")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Aggregate throughput: {total_tokens / total_time:.1f} tok/s")
    print(f"{'=' * 60}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM client demo")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    async with VLLMClient(base_url=args.base_url, model=args.model) as client:
        await demo_streaming(client)
        await demo_batch(client)
        await demo_concurrent(client)


if __name__ == "__main__":
    asyncio.run(main())
