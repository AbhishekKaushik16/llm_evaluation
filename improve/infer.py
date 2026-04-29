#!/usr/bin/env python3
"""Inference with optimized prompts and self-consistency voting.

Runs the optimized prompts against the vLLM server, collects answers,
applies self-consistency (majority voting), and evaluates accuracy.

Usage:
    python infer.py --base-url http://localhost:8000 --model Qwen/Qwen2.5-3B-Instruct
    python infer.py --strategy combined --self-consistency-k 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import aiohttp
from aiohttp import ClientResponseError
from aiohttp.client_exceptions import ClientError, ServerDisconnectedError

import numpy as np
from scipy import stats
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from improve.optimize_prompt import normalize_answer

PROMPTS_DIR = Path(__file__).parent / "prompts"
RESULTS_DIR = Path(__file__).parent / "results"


async def call_completion(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int | None,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }
    if seed is not None:
        payload["seed"] = seed

    # Transient disconnects under load are common on long runs; retry before failing.
    last_err: BaseException | None = None
    for attempt in range(5):
        try:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["text"]
        except (ServerDisconnectedError, ClientError, ClientResponseError, asyncio.TimeoutError) as e:
            last_err = e
            if attempt < 4:
                await asyncio.sleep(1.0 * (2**attempt))
                continue
            raise
    assert last_err is not None
    raise last_err


async def run_inference(
    url: str,
    model: str,
    prompts: list[dict],
    max_tokens: int = 150,
    temperature: float = 0.0,
    seed: int = 42,
    self_consistency_k: int = 1,
    concurrency: int = 8,
) -> list[dict]:
    """Run inference on all prompts with optional self-consistency."""
    timeout = aiohttp.ClientTimeout(total=300)
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def process_one(item: dict) -> dict:
            async with semaphore:
                try:
                    answers = []
                    raw_outputs = []

                    for k_idx in range(self_consistency_k):
                        t = temperature if self_consistency_k > 1 else 0.0
                        s = (seed + k_idx) if seed is not None else None

                        output = await call_completion(
                            session, url, model, item["prompt"],
                            max_tokens, t, s,
                        )
                        raw_outputs.append(output)
                        ans = normalize_answer(output)
                        if ans:
                            answers.append(ans)

                    if self_consistency_k > 1 and answers:
                        # Majority vote
                        counter = Counter(answers)
                        final_answer = counter.most_common(1)[0][0]
                    else:
                        final_answer = answers[0] if answers else None

                    return {
                        "id": item["id"],
                        "answerKey": item["answerKey"],
                        "predicted": final_answer,
                        "raw_outputs": raw_outputs,
                        "all_answers": answers,
                        "correct": final_answer == item["answerKey"],
                        "strategy": item["strategy"],
                    }
                except Exception as e:
                    # Never raise out of a task: one disconnect must not abort the whole batch
                    # (that would exit async with session early and cascade Session is closed).
                    return {
                        "id": item["id"],
                        "answerKey": item["answerKey"],
                        "predicted": None,
                        "raw_outputs": [f"[client error after retries] {type(e).__name__}: {e}"],
                        "all_answers": [],
                        "correct": False,
                        "strategy": item["strategy"],
                        "error": repr(e),
                    }

        tasks = [asyncio.create_task(process_one(item)) for item in prompts]

        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Running inference",
        ):
            result = await coro
            results.append(result)

    return results


def compute_accuracy_with_ci(results: list[dict], confidence: float = 0.95) -> dict:
    """Compute accuracy with bootstrap confidence interval."""
    correct = np.array([1 if r["correct"] else 0 for r in results])
    accuracy = float(np.mean(correct))

    # Bootstrap CI
    n_bootstrap = 1000
    rng = np.random.default_rng(42)
    bootstrap_accs = []
    for _ in range(n_bootstrap):
        sample = rng.choice(correct, size=len(correct), replace=True)
        bootstrap_accs.append(float(np.mean(sample)))

    alpha = 1 - confidence
    ci_low = float(np.percentile(bootstrap_accs, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_accs, 100 * (1 - alpha / 2)))

    return {
        "accuracy": accuracy,
        "accuracy_pct": accuracy * 100,
        "ci_low": ci_low * 100,
        "ci_high": ci_high * 100,
        "n": len(correct),
        "n_correct": int(np.sum(correct)),
    }


def significance_test(baseline_results: list[dict], improved_results: list[dict]) -> dict:
    """McNemar's test for comparing paired binary outcomes."""
    baseline_correct = {r["id"]: r["correct"] for r in baseline_results}
    improved_correct = {r["id"]: r["correct"] for r in improved_results}

    common_ids = set(baseline_correct.keys()) & set(improved_correct.keys())

    # Contingency: b = baseline wrong, improved right; c = baseline right, improved wrong
    b = sum(1 for qid in common_ids if not baseline_correct[qid] and improved_correct[qid])
    c = sum(1 for qid in common_ids if baseline_correct[qid] and not improved_correct[qid])

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    # McNemar's test with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {"statistic": float(statistic), "p_value": float(p_value), "b": b, "c": c}


def print_examples(
    baseline_results: list[dict],
    improved_results: list[dict],
    n: int = 10,
) -> str:
    """Generate before/after comparison examples."""
    baseline_map = {r["id"]: r for r in baseline_results}
    improved_map = {r["id"]: r for r in improved_results}

    # Find examples where improved got it right but baseline didn't
    flipped = [
        qid for qid in baseline_map
        if qid in improved_map
        and not baseline_map[qid]["correct"]
        and improved_map[qid]["correct"]
    ]

    lines = []
    for qid in flipped[:n]:
        b = baseline_map[qid]
        imp = improved_map[qid]
        lines.append(f"\n### Question ID: {qid}")
        lines.append(f"**Correct Answer**: {b['answerKey']}")
        lines.append(f"**Baseline predicted**: {b['predicted']} ({'correct' if b['correct'] else 'WRONG'})")
        lines.append(f"**Improved predicted**: {imp['predicted']} ({'correct' if imp['correct'] else 'WRONG'})")
        lines.append(f"**Baseline output** (truncated): {b['raw_outputs'][0][:200]}")
        lines.append(f"**Improved output** (truncated): {imp['raw_outputs'][0][:200]}")
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optimized inference for ARC-Challenge")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["all", "baseline", "instruction", "few_shot", "cot", "combined"])
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--self-consistency-k", type=int, default=1,
                        help="Number of samples for self-consistency (>1 enables majority voting)")
    parser.add_argument("--sc-temperature", type=float, default=0.7,
                        help="Temperature for self-consistency sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of prompts per strategy (None = full ARC-Challenge test).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    completions_url = f"{args.base_url}/v1/completions"

    strategies = (
        ["baseline", "instruction", "few_shot", "cot", "combined"]
        if args.strategy == "all"
        else [args.strategy]
    )

    all_strategy_results = {}

    for strategy_name in strategies:
        prompt_file = PROMPTS_DIR / f"prompts_{strategy_name}.jsonl"
        if not prompt_file.exists():
            print(f"Prompt file not found: {prompt_file}. Run optimize_prompt.py first.")
            continue

        prompts = []
        with open(prompt_file) as f:
            for line in f:
                prompts.append(json.loads(line))

        if args.limit is not None:
            prompts = prompts[: args.limit]

        # Respect --self-consistency-k for any strategy (e.g. few_shot + SC).
        # `make improve` / eval.sh use defaults (k=1) for the greedy sweep, then
        # invoke combined + SC in a second pass with k>1.
        sc_k = args.self_consistency_k
        temp = args.sc_temperature if sc_k > 1 else 0.0

        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name} (k={sc_k}, temp={temp})")
        print(f"{'='*60}")

        results = await run_inference(
            completions_url, args.model, prompts,
            max_tokens=args.max_tokens,
            temperature=temp,
            seed=args.seed,
            self_consistency_k=sc_k,
            concurrency=args.concurrency,
        )

        accuracy = compute_accuracy_with_ci(results)
        all_strategy_results[strategy_name] = {
            "results": results,
            "accuracy": accuracy,
        }

        print(f"\nAccuracy: {accuracy['accuracy_pct']:.2f}% "
              f"[{accuracy['ci_low']:.2f}%, {accuracy['ci_high']:.2f}%] "
              f"({accuracy['n_correct']}/{accuracy['n']})")

        suffix = f"_sc{sc_k}" if sc_k > 1 else ""
        results_file = RESULTS_DIR / f"results_{strategy_name}{suffix}.json"
        with open(results_file, "w") as f:
            json.dump({
                "strategy": strategy_name,
                "accuracy": accuracy,
                "self_consistency_k": sc_k,
                "temperature": temp,
                "seed": args.seed,
                "model": args.model,
                "results": results,
            }, f, indent=2)
        print(f"Saved to {results_file}")

    # Print comparison table
    if len(all_strategy_results) > 1:
        print(f"\n{'='*72}")
        print("ABLATION STUDY - ARC-Challenge")
        print(f"{'='*72}")
        print(f"{'Strategy':<20} {'Accuracy':>10} {'95% CI':>20} {'N Correct':>12}")
        print("-" * 64)

        baseline_results = None
        for name in ["baseline", "instruction", "few_shot", "cot", "combined"]:
            if name not in all_strategy_results:
                continue
            acc = all_strategy_results[name]["accuracy"]
            print(f"{name:<20} {acc['accuracy_pct']:>10.2f}% "
                  f"[{acc['ci_low']:>7.2f}%, {acc['ci_high']:>7.2f}%] "
                  f"{acc['n_correct']:>8}/{acc['n']}")

            if name == "baseline":
                baseline_results = all_strategy_results[name]["results"]

        # Significance tests
        if baseline_results:
            print(f"\n{'Strategy':<20} {'Lift':>10} {'p-value':>10} {'Significant':>12}")
            print("-" * 54)
            baseline_acc = all_strategy_results["baseline"]["accuracy"]["accuracy_pct"]
            for name in ["instruction", "few_shot", "cot", "combined"]:
                if name not in all_strategy_results:
                    continue
                improved_acc = all_strategy_results[name]["accuracy"]["accuracy_pct"]
                sig = significance_test(baseline_results, all_strategy_results[name]["results"])
                lift = improved_acc - baseline_acc
                is_sig = "YES" if sig["p_value"] < 0.05 else "NO"
                print(f"{name:<20} {lift:>+10.2f} {sig['p_value']:>10.4f} {is_sig:>12}")

        # Generate examples
        if baseline_results and "combined" in all_strategy_results:
            examples = print_examples(
                baseline_results, all_strategy_results["combined"]["results"], n=12,
            )
            examples_file = RESULTS_DIR / "before_after_examples.md"
            with open(examples_file, "w") as f:
                f.write("# Before/After Examples (Baseline vs Combined Strategy)\n\n")
                f.write(examples)
            print(f"\nExamples saved to {examples_file}")


if __name__ == "__main__":
    asyncio.run(main())
