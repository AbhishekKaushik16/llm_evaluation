#!/usr/bin/env python3
"""Generation-based multiple-choice evaluator.

Workaround for inference servers (e.g. some vllm-metal builds) that do not
return logprobs reliably. Instead of scoring each candidate via
loglikelihood, we render the choices as A/B/C/D and ask the model to
generate the answer letter, then exact-match against the gold answer.

Supported task formats:
  - custom_commonsense:  local JSONL with {question, choices[4], answer}
  - mmlu:                HF dataset 'cais/mmlu' (per-subject configs)
  - hellaswag:           HF dataset 'hellaswag' (4-way endings)

Usage:
    from eval_runner.gen_mc_eval import evaluate_task
    result = evaluate_task("custom_commonsense", base_url, model)
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

LETTER_CHOICES = ["A", "B", "C", "D"]
_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Prompt rendering & answer parsing
# ---------------------------------------------------------------------------


def render_prompt(question: str, choices: list[str]) -> str:
    if len(choices) != 4:
        raise ValueError(f"Expected 4 choices, got {len(choices)}")
    return (
        f"Question: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:"
    )


def parse_answer_letter(text: str) -> str | None:
    """Extract the predicted A/B/C/D from generated text.

    Strategy: take the first A-D letter that appears, ignoring leading
    whitespace, punctuation, or words like 'The answer is'.
    """
    if not text:
        return None
    stripped = text.strip().upper()
    if stripped and stripped[0] in LETTER_CHOICES:
        return stripped[0]
    m = _LETTER_RE.search(stripped)
    return m.group(1).upper() if m else None


def normalize_gold(answer: Any) -> str:
    """Coerce a dataset's gold answer into a single 'A'/'B'/'C'/'D' letter."""
    if isinstance(answer, int):
        if 0 <= answer < 4:
            return LETTER_CHOICES[answer]
        raise ValueError(f"Integer answer out of range: {answer}")
    if isinstance(answer, str):
        s = answer.strip().upper()
        if s and s[0] in LETTER_CHOICES:
            return s[0]
    raise ValueError(f"Could not normalize gold answer: {answer!r}")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_custom_commonsense() -> list[dict]:
    path = Path(__file__).parent / "custom_task" / "data.jsonl"
    samples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_mmlu(limit: int | None = None) -> list[dict]:
    """Load MMLU 'all' configuration via HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    samples = []
    for ex in ds:
        samples.append({
            "question": ex["question"],
            "choices": ex["choices"],
            "answer": ex["answer"],  # int 0-3
            "subject": ex.get("subject"),
        })
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_hellaswag(limit: int | None = None) -> list[dict]:
    """Load HellaSwag validation split. Each item has 4 candidate endings;
    we treat them as A/B/C/D options."""
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    samples = []
    for ex in ds:
        ctx = ex.get("ctx") or ex.get("ctx_a", "")
        # Question is the context; choices are the 4 endings.
        endings = ex.get("endings", [])
        if len(endings) != 4:
            continue
        samples.append({
            "question": ctx.strip(),
            "choices": [e.strip() for e in endings],
            "answer": int(ex["label"]),  # "0".."3"
        })
        if limit is not None and len(samples) >= limit:
            break
    return samples


_LOADERS: dict[str, Callable[..., list[dict]]] = {
    "custom_commonsense": lambda limit=None: (
        load_custom_commonsense()[:limit] if limit is not None else load_custom_commonsense()
    ),
    "mmlu": load_mmlu,
    "hellaswag": load_hellaswag,
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _post_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    session: requests.Session,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stop": ["\n"],
    }
    resp = session.post(base_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["text"]


def evaluate_samples(
    samples: list[dict],
    *,
    base_url: str,
    model: str,
    max_tokens: int = 5,
    temperature: float = 0.0,
    concurrency: int = 8,
    timeout: float = 120.0,
    progress: bool = True,
) -> dict:
    """Run generation-based MC scoring over a list of samples.

    Each sample must have keys: question (str), choices (list[str] of len 4),
    answer (int 0-3 or letter A-D).

    Returns a dict with aggregate metrics (`acc`) and per-sample details.
    """
    n = len(samples)
    correct = 0
    parseable = 0
    failures: list[dict] = []
    per_sample: list[dict] = [None] * n  # type: ignore[list-item]
    start = time.time()

    session = requests.Session()

    def _run_one(i: int, s: dict) -> tuple[int, dict]:
        prompt = render_prompt(s["question"], s["choices"])
        gold = normalize_gold(s["answer"])
        try:
            text = _post_completion(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                session=session,
            )
        except Exception as e:
            return i, {"error": str(e), "gold": gold, "pred": None, "correct": False}
        pred = parse_answer_letter(text)
        return i, {
            "gold": gold,
            "pred": pred,
            "correct": pred == gold,
            "raw": text,
        }

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = [pool.submit(_run_one, i, s) for i, s in enumerate(samples)]
        for fut in as_completed(futs):
            i, info = fut.result()
            per_sample[i] = info
            if "error" in info:
                failures.append({"index": i, **info})
            else:
                if info["pred"] is not None:
                    parseable += 1
                if info["correct"]:
                    correct += 1
            completed += 1
            if progress and (completed % 25 == 0 or completed == n):
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"  [{completed}/{n}] acc={correct/max(completed,1):.3f} "
                    f"parseable={parseable/max(completed,1):.3f} "
                    f"errors={len(failures)} ({rate:.1f} req/s)"
                )

    acc = correct / n if n else 0.0
    return {
        "acc": acc,
        "acc_norm": acc,
        "n_samples": n,
        "n_correct": correct,
        "n_parseable": parseable,
        "n_errors": len(failures),
        "elapsed_seconds": time.time() - start,
        "failures": failures[:10],
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def evaluate_task(
    task: str,
    *,
    base_url: str,
    model: str,
    limit: int | None = None,
    max_tokens: int = 5,
    temperature: float = 0.0,
    concurrency: int = 8,
) -> dict:
    """Evaluate a named MC task using generation-based scoring.

    Returns a dict with shape compatible with lm-eval's per-task results:
        {"results": {<task>: {"acc,none": float, ...}}, "n-shot": {...}}
    """
    if task not in _LOADERS:
        raise ValueError(
            f"Unsupported task '{task}' for generation-based MC eval. "
            f"Supported: {sorted(_LOADERS.keys())}"
        )
    print(f"Loading task '{task}'...")
    samples = _LOADERS[task](limit=limit) if limit is not None else _LOADERS[task]()
    print(f"  loaded {len(samples)} samples")

    metrics = evaluate_samples(
        samples,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        concurrency=concurrency,
    )

    return {
        "results": {
            task: {
                "acc,none": metrics["acc"],
                "acc_norm,none": metrics["acc"],
                "n_samples": metrics["n_samples"],
                "n_correct": metrics["n_correct"],
                "n_parseable": metrics["n_parseable"],
                "n_errors": metrics["n_errors"],
                "elapsed_seconds": metrics["elapsed_seconds"],
            }
        },
        "n-shot": {task: 0},
        "versions": {task: "gen-1.0"},
        "config": {"strategy": "generation", "scoring": "exact-match-letter"},
        "_failures": metrics["failures"],
    }
