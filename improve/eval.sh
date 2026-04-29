#!/usr/bin/env bash
# Evaluation pipeline: baseline vs improved ARC-Challenge scores.
#
# Usage:
#   bash eval.sh [BASE_URL] [MODEL]
#
# This script:
#   1. Prepares data and embedding index (if not already done)
#   2. Builds optimized prompts for all strategies
#   3. Runs inference for all strategies (including self-consistency)
#   4. Produces an ablation table and before/after examples

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
MODEL="${2:-Qwen/Qwen2.5-3B-Instruct}"
LIMIT="${LIMIT:-}"   # optional: cap prompts per strategy (e.g. LIMIT=100 bash eval.sh)
LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then LIMIT_FLAG="--limit $LIMIT"; fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${VENV_PYTHON:-${HOME}/.venv-vllm-metal/bin/python}"

echo "========================================"
echo "ARC-Challenge Improvement Pipeline"
echo "========================================"
echo "Server:  $BASE_URL"
echo "Model:   $MODEL"
echo ""

# Step 1: Prepare data
if [ ! -f "$SCRIPT_DIR/index/train_embeddings.npy" ]; then
    echo "--- Step 1: Preparing data and building index ---"
    "$PYTHON" "$SCRIPT_DIR/prepare_data.py"
else
    echo "--- Step 1: Data already prepared (skipping) ---"
fi

# Step 2: Build optimized prompts
echo ""
echo "--- Step 2: Building optimized prompts for all strategies ---"
"$PYTHON" "$SCRIPT_DIR/optimize_prompt.py" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --strategy all

# Step 3: Run inference for all strategies
echo ""
echo "--- Step 3: Running inference (baseline, single-strategy, combined) ---"

# Baseline + single strategies (greedy decoding)
"$PYTHON" "$SCRIPT_DIR/infer.py" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --strategy all \
    --max-tokens 150 \
    --seed 42 \
    --concurrency 8 \
    $LIMIT_FLAG

# Combined with self-consistency (k=5)
echo ""
echo "--- Step 4: Running combined strategy with self-consistency (k=5) ---"
"$PYTHON" "$SCRIPT_DIR/infer.py" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --strategy combined \
    --max-tokens 150 \
    --self-consistency-k 5 \
    --sc-temperature 0.7 \
    --seed 42 \
    --concurrency 8 \
    $LIMIT_FLAG

echo ""
echo "========================================"
echo "Pipeline complete. Results in: $SCRIPT_DIR/results/"
echo "========================================"
