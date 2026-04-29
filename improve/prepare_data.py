#!/usr/bin/env python3
"""Prepare ARC-Challenge data and build a semantic embedding index for few-shot retrieval.

Downloads the ARC-Challenge dataset, embeds all training questions using
sentence-transformers, and saves a FAISS index for fast nearest-neighbor lookup.

Usage:
    python prepare_data.py
"""

import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ARC-Challenge dataset...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    # Save splits to JSONL
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            continue
        output_file = DATA_DIR / f"arc_challenge_{split_name}.jsonl"
        with open(output_file, "w") as f:
            for item in ds[split_name]:
                record = {
                    "id": item["id"],
                    "question": item["question"],
                    "choices": {
                        "text": item["choices"]["text"],
                        "label": item["choices"]["label"],
                    },
                    "answerKey": item["answerKey"],
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Saved {len(ds[split_name])} examples to {output_file}")

    # Build embedding index from training set
    print("\nBuilding embedding index for few-shot retrieval...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    train_questions = [item["question"] for item in ds["train"]]
    train_records = []
    for item in ds["train"]:
        train_records.append({
            "id": item["id"],
            "question": item["question"],
            "choices": {
                "text": item["choices"]["text"],
                "label": item["choices"]["label"],
            },
            "answerKey": item["answerKey"],
        })

    print(f"  Encoding {len(train_questions)} training questions...")
    embeddings = model.encode(train_questions, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Save embeddings and metadata
    np.save(INDEX_DIR / "train_embeddings.npy", embeddings)
    with open(INDEX_DIR / "train_records.json", "w") as f:
        json.dump(train_records, f)

    # Build FAISS index
    try:
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
        index.add(embeddings)
        faiss.write_index(index, str(INDEX_DIR / "train.faiss"))
        print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    except ImportError:
        print("  FAISS not available; will fall back to numpy cosine similarity at query time")

    print("\nData preparation complete.")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Index directory: {INDEX_DIR}")


if __name__ == "__main__":
    main()
