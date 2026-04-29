# Guardrails & Determinism

## What Was Tested

### 1. Deterministic Generation
- **Method**: Send the same prompt 5 times with `temperature=0`, `top_p=1`, `seed=42`
- **Assertion**: All responses are byte-identical
- **Scope**: Tested with a single repeated prompt and across 3 varied prompts

### 2. Output Schema Validation
- **MCQ format**: Verify model outputs match `^[A-D]` regex for multiple-choice questions
- **JSON format**: Verify model can produce valid JSON with expected keys and types
- **Purpose**: Ensures outputs from the custom benchmark can be parsed reliably

### 3. Edge Cases
- **Empty prompt**: Server should return a response without crashing
- **Long prompt** (~3000 tokens): Server should handle prompts near the context window limit
- **Prompt injection**: Basic adversarial prompt that attempts to override instructions

## Where Nondeterminism May Persist

Even with `temperature=0` and a fixed `seed`, nondeterminism can arise from:

1. **Floating-point non-associativity**: Parallel GPU reductions (e.g., in attention softmax or layer norms) may sum values in different orders depending on thread scheduling, producing slightly different floating-point results. This is inherent to parallel computation.

2. **Batch composition effects**: When using continuous batching, the specific set of requests being processed together can affect the internal scheduling and padding. A prompt processed alone vs. in a batch may see different numerical paths.

3. **KV cache state**: In paged attention, the physical memory layout of KV cache pages depends on prior requests. While logically equivalent, different page arrangements can lead to different memory access patterns.

4. **Non-deterministic kernels**: Some CUDA/Metal kernels (e.g., `atomicAdd` in certain attention implementations) are inherently non-deterministic. vLLM's `--seed` flag helps but cannot eliminate all sources.

5. **Tokenizer edge cases**: Certain Unicode sequences or whitespace patterns may tokenize differently depending on normalization, though this is rare with standard prompts.

## Mitigation Strategies

- Always set `temperature=0`, `top_p=1`, and explicit `seed` for reproducible runs
- Use the prompt cache (SQLite) in `eval_runner/vllm_model.py` to ensure identical inputs always return the same cached output across evaluation runs
- For critical comparisons, run multiple trials and report variance
- Pin the model revision and vLLM version in production
