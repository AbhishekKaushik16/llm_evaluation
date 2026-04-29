#!/usr/bin/env python3
"""Evaluation runner: runs lm-evaluation-harness benchmarks against the vLLM server.

Usage:
    python run_eval.py --base-url http://localhost:8000/v1/completions --tasks mmlu hellaswag
    python run_eval.py --tasks custom_commonsense --limit 50
"""

import argparse
import json
import sys
import time
import ssl
from pathlib import Path
from datetime import datetime

import lm_eval
from lm_eval.tasks import TaskManager

# Ensure the wrapper is registered
sys.path.insert(0, str(Path(__file__).parent))
import vllm_model  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LM evaluation benchmarks")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "hellaswag"])
    parser.add_argument("--output-dir", type=str, default="eval_runner/results")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--apply-chat-template", action="store_true", default=False)
    parser.add_argument("--retries", type=int, default=3, help="Retry count for transient HF/network errors")
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=10.0,
        help="Base backoff (seconds) between retries; actual wait = base * attempt",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run tasks even if a per-task result file already exists",
    )
    parser.add_argument(
        "--strategy",
        choices=["loglikelihood", "generation"],
        default="loglikelihood",
        help=(
            "Scoring strategy. 'loglikelihood' uses lm-eval-harness (requires "
            "the server to support logprobs). 'generation' bypasses logprobs "
            "and uses a generation-based MC scorer (works with vllm-metal "
            "builds where logprobs are broken)."
        ),
    )
    parser.add_argument(
        "--gen-concurrency",
        type=int,
        default=8,
        help="Concurrency for generation-based MC scoring",
    )
    parser.add_argument(
        "--gen-max-tokens",
        type=int,
        default=5,
        help="max_tokens used for generation-based MC scoring",
    )
    return parser.parse_args()


def format_results_table(results: dict) -> str:
    lines = []
    lines.append(f"{'Task':<30} {'Metric':<20} {'Value':>10} {'Stderr':>10}")
    lines.append("-" * 72)

    task_results = results.get("results", {})
    for task_name, metrics in sorted(task_results.items()):
        for metric_name, value in sorted(metrics.items()):
            if metric_name.endswith(",none"):
                display_name = metric_name.replace(",none", "")
            elif metric_name.startswith("alias"):
                continue
            else:
                display_name = metric_name

            stderr_key = metric_name + "_stderr,none" if not metric_name.endswith(",none") else metric_name.replace(",none", "_stderr,none")
            stderr = metrics.get(stderr_key, "")
            if isinstance(stderr, (int, float)):
                stderr_str = f"{stderr:.4f}"
            else:
                stderr_str = str(stderr) if stderr else "N/A"

            if isinstance(value, float):
                lines.append(f"{task_name:<30} {display_name:<20} {value:>10.4f} {stderr_str:>10}")
            elif isinstance(value, (int, str)):
                lines.append(f"{task_name:<30} {display_name:<20} {str(value):>10} {stderr_str:>10}")

    return "\n".join(lines)


def _slug(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def _task_result_path(output_dir: Path, model: str, task: str, strategy: str) -> Path:
    # Include strategy in the cache key so loglikelihood and generation-based
    # results don't collide for the same (model, task).
    return output_dir / f"task_{_slug(model)}__{_slug(task)}__{strategy}.json"


def _retry_exception_classes() -> tuple[type[BaseException], ...]:
    """Genuinely transient errors worth retrying. Excludes 4xx/5xx HTTP
    responses, which are deterministic and won't fix themselves."""
    classes: list[type[BaseException]] = [ssl.SSLError, ConnectionError, TimeoutError]
    try:
        import httpx

        classes.extend([
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ])
    except Exception:
        pass
    try:
        import requests

        classes.extend([
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ])
    except Exception:
        pass
    return tuple(classes)


def _evaluate_one_task_lm_eval(
    *,
    task: str,
    model_args_str: str,
    task_manager: TaskManager,
    args: argparse.Namespace,
    retry_exceptions: tuple[type[BaseException], ...],
) -> dict | None:
    """Run a single task via lm-evaluation-harness (loglikelihood path)."""
    last_err: BaseException | None = None
    for attempt in range(1, max(args.retries, 1) + 1):
        try:
            return lm_eval.simple_evaluate(
                model="cached_vllm",
                model_args=model_args_str,
                tasks=[task],
                num_fewshot=args.num_fewshot,
                limit=args.limit,
                batch_size=args.batch_size,
                task_manager=task_manager,
                apply_chat_template=args.apply_chat_template,
            )
        except retry_exceptions as e:
            last_err = e
            if attempt >= args.retries:
                break
            wait_s = args.retry_backoff_seconds * attempt
            print(f"[WARN] transient error on task '{task}': {e}")
            print(f"[WARN] retrying in {wait_s:.1f}s (attempt {attempt}/{args.retries})...")
            time.sleep(wait_s)
        except Exception as e:
            last_err = e
            print(f"[ERROR] task '{task}' failed with non-retryable error: {e}")
            break

    if last_err is not None:
        print(f"[ERROR] giving up on task '{task}' after {args.retries} attempt(s): {last_err}")
    return None


def _evaluate_one_task_generation(
    *,
    task: str,
    args: argparse.Namespace,
) -> dict | None:
    """Run a single task via the generation-based MC scorer."""
    from gen_mc_eval import evaluate_task as gen_evaluate_task

    try:
        return gen_evaluate_task(
            task,
            base_url=args.base_url,
            model=args.model,
            limit=args.limit,
            max_tokens=args.gen_max_tokens,
            concurrency=args.gen_concurrency,
        )
    except ValueError as e:
        print(f"[ERROR] task '{task}' not supported by generation strategy: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] task '{task}' failed during generation eval: {e}")
        return None


def _evaluate_one_task(
    *,
    task: str,
    model_args_str: str,
    task_manager: TaskManager,
    args: argparse.Namespace,
    retry_exceptions: tuple[type[BaseException], ...],
) -> dict | None:
    if args.strategy == "generation":
        return _evaluate_one_task_generation(task=task, args=args)
    return _evaluate_one_task_lm_eval(
        task=task,
        model_args_str=model_args_str,
        task_manager=task_manager,
        args=args,
        retry_exceptions=retry_exceptions,
    )


def main() -> None:
    args = parse_args()

    # Register custom task directory for our custom benchmark
    custom_task_dir = Path(__file__).parent / "custom_task"
    include_path = str(custom_task_dir) if custom_task_dir.exists() else None
    task_manager = TaskManager(include_path=include_path)

    model_args_str = f"base_url={args.base_url},model={args.model}"

    print(f"Evaluating model: {args.model}")
    print(f"Tasks: {args.tasks}")
    print(f"Server: {args.base_url}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retry_exceptions = _retry_exception_classes()

    # Run each task independently so a failure on one doesn't blow away
    # progress on the others. Per-task results are persisted immediately and
    # reused on subsequent invocations unless --force is passed.
    aggregated_results: dict[str, dict] = {}
    aggregated_versions: dict[str, str] = {}
    aggregated_nshot: dict[str, int] = {}
    failed_tasks: list[str] = []

    for task in args.tasks:
        task_file = _task_result_path(output_dir, args.model, task, args.strategy)
        if task_file.exists() and not args.force:
            print(f"[SKIP] task '{task}' already has results at {task_file} (use --force to redo)")
            try:
                cached = json.loads(task_file.read_text())
                aggregated_results.update(cached.get("results", {}))
                aggregated_versions.update(cached.get("versions", {}))
                aggregated_nshot.update(cached.get("n-shot", {}))
                continue
            except Exception as e:
                print(f"[WARN] could not parse cached result {task_file}: {e}; re-running")

        print(f"\n[RUN] evaluating task '{task}' ...")
        task_results = _evaluate_one_task(
            task=task,
            model_args_str=model_args_str,
            task_manager=task_manager,
            args=args,
            retry_exceptions=retry_exceptions,
        )

        if task_results is None:
            failed_tasks.append(task)
            continue

        per_task = {
            "task": task,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "results": task_results.get("results", {}),
            "versions": task_results.get("versions", {}),
            "n-shot": task_results.get("n-shot", {}),
            "config": task_results.get("config", {}),
        }
        task_file.write_text(json.dumps(per_task, indent=2, default=str))
        print(f"[OK] task '{task}' results saved to {task_file}")

        aggregated_results.update(per_task["results"])
        aggregated_versions.update(per_task["versions"])
        aggregated_nshot.update(per_task["n-shot"])

    if not aggregated_results:
        print("\n[ERROR] no tasks completed successfully.")
        if failed_tasks:
            print(f"Failed tasks: {failed_tasks}")
        sys.exit(1)

    # Print summary table over whatever did succeed
    results = {
        "results": aggregated_results,
        "versions": aggregated_versions,
        "n-shot": aggregated_nshot,
    }
    table = format_results_table(results)
    print("\n" + "=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)
    print(table)
    print("=" * 72)
    if failed_tasks:
        print(f"\n[WARN] tasks that failed and were skipped: {failed_tasks}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_slug = "_".join(args.tasks)

    results_file = output_dir / f"results_{task_slug}_{timestamp}.json"
    serializable = {
        "results": aggregated_results,
        "versions": aggregated_versions,
        "n-shot": aggregated_nshot,
        "timestamp": timestamp,
        "model": args.model,
        "tasks": args.tasks,
        "failed_tasks": failed_tasks,
    }
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nAggregate results saved to: {results_file}")

    summary_file = output_dir / f"summary_{task_slug}_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Tasks: {', '.join(args.tasks)}\n")
        if failed_tasks:
            f.write(f"Failed: {', '.join(failed_tasks)}\n")
        f.write("\n")
        f.write(table)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
