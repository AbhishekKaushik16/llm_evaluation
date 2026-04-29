#!/usr/bin/env python3
"""Prompt optimization strategies for ARC-Challenge.

Implements:
  1. Template rewriting (structured instruction prompts)
  2. Automatic few-shot selection via semantic similarity
  3. Chain-of-thought prompting
  4. Output normalization

Usage:
    python optimize_prompt.py --base-url http://localhost:8000 --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent / "index"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "prompts"


def load_train_index() -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load(INDEX_DIR / "train_embeddings.npy")
    with open(INDEX_DIR / "train_records.json") as f:
        records = json.load(f)
    return embeddings, records


def retrieve_similar(
    query_embedding: np.ndarray,
    train_embeddings: np.ndarray,
    train_records: list[dict],
    k: int = 3,
) -> list[dict]:
    """Retrieve top-k most similar training examples using cosine similarity."""
    try:
        import faiss
        index_path = INDEX_DIR / "train.faiss"
        if index_path.exists():
            index = faiss.read_index(str(index_path))
            query = query_embedding.reshape(1, -1).astype(np.float32)
            _, indices = index.search(query, k)
            return [train_records[i] for i in indices[0] if i >= 0]
    except ImportError:
        pass

    # Fallback: numpy cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    similarities = train_embeddings @ query_norm
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    return [train_records[i] for i in top_k_idx]


def format_arc_question(record: dict, include_answer: bool = False) -> str:
    """Format a single ARC question with choices."""
    lines = [f"Question: {record['question']}"]
    for label, text in zip(record["choices"]["label"], record["choices"]["text"]):
        lines.append(f"  {label}. {text}")
    if include_answer:
        lines.append(f"Answer: {record['answerKey']}")
    return "\n".join(lines)


class PromptStrategy:
    """Base class for prompt optimization strategies."""

    name: str = "base"

    def format_prompt(self, record: dict, **kwargs) -> str:
        raise NotImplementedError


class BaselineStrategy(PromptStrategy):
    """Minimal zero-shot prompt (lm-eval-harness default style)."""

    name = "baseline"

    def format_prompt(self, record: dict, **kwargs) -> str:
        return format_arc_question(record) + "\nAnswer:"


class InstructionStrategy(PromptStrategy):
    """Structured instruction prompt with task description."""

    name = "instruction"

    def format_prompt(self, record: dict, **kwargs) -> str:
        return (
            "You are an expert at answering science questions. "
            "Read the question carefully and select the single best answer from the choices provided. "
            "Respond with ONLY the letter of the correct answer.\n\n"
            + format_arc_question(record)
            + "\nAnswer:"
        )


class FewShotStrategy(PromptStrategy):
    """Few-shot prompting with semantically similar examples."""

    name = "few_shot"

    def __init__(self, k: int = 3):
        self.k = k

    def format_prompt(
        self,
        record: dict,
        similar_examples: list[dict] | None = None,
        **kwargs,
    ) -> str:
        parts = [
            "Answer each science question by selecting the correct letter.\n"
        ]
        if similar_examples:
            for ex in similar_examples[:self.k]:
                parts.append(format_arc_question(ex, include_answer=True))
                parts.append("")
        parts.append(format_arc_question(record))
        parts.append("Answer:")
        return "\n".join(parts)


class ChainOfThoughtStrategy(PromptStrategy):
    """Chain-of-thought prompting: encourage step-by-step reasoning."""

    name = "cot"

    def __init__(self, k: int = 3):
        self.k = k

    def format_prompt(
        self,
        record: dict,
        similar_examples: list[dict] | None = None,
        **kwargs,
    ) -> str:
        parts = [
            "You are an expert at answering science questions. "
            "For each question, think step by step before giving your final answer.\n"
        ]
        if similar_examples:
            for ex in similar_examples[:self.k]:
                parts.append(format_arc_question(ex, include_answer=False))
                parts.append(f"Let's think step by step. The correct answer is {ex['answerKey']}.\n")
        parts.append(format_arc_question(record))
        parts.append("Let's think step by step.")
        return "\n".join(parts)


class CombinedStrategy(PromptStrategy):
    """Combines instruction + few-shot + chain-of-thought."""

    name = "combined"

    def __init__(self, k: int = 3):
        self.k = k

    def format_prompt(
        self,
        record: dict,
        similar_examples: list[dict] | None = None,
        **kwargs,
    ) -> str:
        parts = [
            "You are an expert science tutor. For each question below, "
            "reason through the problem step by step, then state the correct answer letter.\n"
        ]
        if similar_examples:
            for ex in similar_examples[:self.k]:
                parts.append(format_arc_question(ex, include_answer=False))
                parts.append(
                    f"Reasoning: This is a science question. Analyzing each option carefully, "
                    f"the correct answer is {ex['answerKey']}.\n"
                )
        parts.append(format_arc_question(record))
        parts.append("Reasoning:")
        return "\n".join(parts)


ALL_STRATEGIES = {
    "baseline": BaselineStrategy(),
    "instruction": InstructionStrategy(),
    "few_shot": FewShotStrategy(k=3),
    "cot": ChainOfThoughtStrategy(k=3),
    "combined": CombinedStrategy(k=3),
}


def normalize_answer(text: str) -> str | None:
    """Extract a single answer letter (A/B/C/D/E) from model output."""
    import re

    text = text.strip()

    # Direct letter match
    m = re.match(r"^([A-E])\b", text)
    if m:
        return m.group(1)

    # "The answer is X" pattern
    m = re.search(r"(?:answer|correct)\s+(?:is|:)\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # Letter followed by period or parenthesis
    m = re.search(r"\b([A-E])[.)]\s", text)
    if m:
        return m.group(1)

    # Last single letter in the text
    letters = re.findall(r"\b([A-E])\b", text)
    if letters:
        return letters[-1]

    return None


def build_optimized_prompts(
    test_records: list[dict],
    strategy: PromptStrategy,
    embed_model: SentenceTransformer | None = None,
    train_embeddings: np.ndarray | None = None,
    train_records: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Build optimized prompts for all test records."""
    results = []
    needs_retrieval = strategy.name in ("few_shot", "cot", "combined")

    for record in test_records:
        similar = None
        if needs_retrieval and embed_model and train_embeddings is not None:
            q_emb = embed_model.encode([record["question"]])[0]
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
            similar = retrieve_similar(q_emb, train_embeddings, train_records, k=3)

        prompt = strategy.format_prompt(record, similar_examples=similar)
        results.append({
            "id": record["id"],
            "prompt": prompt,
            "answerKey": record["answerKey"],
            "strategy": strategy.name,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["all", "baseline", "instruction", "few_shot", "cot", "combined"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_file = DATA_DIR / "arc_challenge_test.jsonl"
    test_records = []
    with open(test_file) as f:
        for line in f:
            test_records.append(json.loads(line))
    print(f"Loaded {len(test_records)} test records")

    # Load training index
    train_embeddings, train_records = load_train_index()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Loaded {len(train_records)} training records with embeddings")

    strategies = (
        list(ALL_STRATEGIES.values()) if args.strategy == "all"
        else [ALL_STRATEGIES[args.strategy]]
    )

    for strategy in strategies:
        print(f"\nBuilding prompts for strategy: {strategy.name}")
        prompts = build_optimized_prompts(
            test_records, strategy, embed_model, train_embeddings, train_records,
        )
        output_file = OUTPUT_DIR / f"prompts_{strategy.name}.jsonl"
        with open(output_file, "w") as f:
            for p in prompts:
                f.write(json.dumps(p) + "\n")
        print(f"  Saved {len(prompts)} prompts to {output_file}")

    print("\nPrompt optimization complete.")


if __name__ == "__main__":
    main()
