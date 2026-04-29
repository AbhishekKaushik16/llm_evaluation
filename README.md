# LLM Evaluation Pipeline

A scaled-down internal LLM evaluation pipeline built on **vLLM** (high-throughput inference) and **lm-evaluation-harness** (standardized benchmarks).

## Quick Start

### Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4) **or** Linux with NVIDIA GPU
- Python 3.11+
- Hugging Face account (only needed for gated models; default model is ungated)

### Installation

```bash
# Install vllm-metal (Apple Silicon)
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate

# Install project dependencies
pip install -r requirements.txt
```

Or use the Makefile:

```bash
make install
```

### Start the Server

The default model is `Qwen/Qwen2.5-3B-Instruct` (ungated, ~6 GB, runs comfortably on Apple Silicon).

```bash
make serve
# Or override with any HuggingFace model id:
make serve MODEL=Qwen/Qwen2.5-7B-Instruct
```

#### Two server profiles

Two backends are wired in. Pick by use case:

| Profile | Command | Speed | logprobs | When to use |
|---|---|---|---|---|
| Metal (default) | `make serve` | Fast (Apple GPU) | Broken (vllm-metal 0.2.0) | `make perf`, `make demo`, `make eval-gen` |
| CPU             | `make serve-cpu` | Slow (~5–15 tok/s) | Working (upstream) | `make eval` (canonical loglikelihood path) |

`make serve-cpu` sets `VLLM_PLUGINS=""` in the child environment, which causes vLLM to skip the metal platform plugin and fall back to its built-in `CpuPlatform`. The OpenAI-compatible API is identical, so all client code is unchanged.

Both profiles bind to `PORT=8000` by default — run **one at a time** unless you also override `PORT`.

### Run the Demo

```bash
make demo
```

## Project Structure

```
├── Makefile                  # One-command targets for all tasks
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── serve/                    # Part A: Serving
│   ├── serve.py              # vLLM server launcher
│   ├── client.py             # Async Python client (streaming, configurable)
│   └── demo.py               # Sample prompt generations
├── eval_runner/              # Part B: Evaluation
│   ├── vllm_model.py         # Cached model wrapper for lm-eval-harness
│   ├── run_eval.py           # Evaluation runner script
│   ├── custom_task/          # Custom JSON benchmark
│   │   ├── task_config.yaml  # lm-eval task configuration
│   │   └── data.jsonl        # Custom benchmark data
│   └── results/              # Benchmark output files
├── perf/                     # Part C: Performance & Scaling
│   ├── load_test.py          # Concurrent load generator
│   ├── metrics.csv           # Raw performance data
│   └── analysis.ipynb        # Plots and commentary
├── guardrails/               # Part D: Guardrails & Determinism
│   ├── validate.py           # Determinism + schema validation
│   └── README.md             # Testing notes
└── improve/                  # Part E: Benchmark Improvement
    ├── prepare_data.py       # Data prep & embedding index
    ├── optimize_prompt.py    # Prompt optimization strategies
    ├── infer.py              # Inference with optimized prompts
    ├── eval.sh               # Baseline vs improved comparison
    └── report.md             # Results report
```

## Part A: Serving

Spins up a vLLM inference server with an OpenAI-compatible API. The async Python client supports streaming token generation, configurable decoding parameters, and concurrent requests.

```bash
make serve   # Start vLLM server
make demo    # Run sample generations
```

## Part B: Evaluation

Integrates the served model with lm-evaluation-harness for standardized benchmarks (MMLU, HellaSwag) plus a custom commonsense reasoning task. Includes SQLite-based prompt caching.

```bash
make eval-gen                              # Reported metric: generation-based MC, fast on metal
make eval-gen TASKS=custom_commonsense     # Just the custom 50-question task
make eval TASKS=custom_commonsense LIMIT=20  # Canonical loglikelihood (CPU server, experimental)
```

### Note on `vllm-metal` and logprobs

The Apple-Silicon `vllm-metal` build (v0.2.0) returns HTTP 500 (`IndexError` inside `_create_completion_logprobs`) for **any** request that asks for `logprobs`, `prompt_logprobs`, or `echo`. Plain text generation works fine.

Root cause is in `vllm_metal/v1/model_runner.py`: every code path that constructs `ModelRunnerOutput` hardcodes `logprobs=None, prompt_logprobs_dict={}`, and `sampling_batch.py` sets `max_num_logprobs=None`. The MLX greedy sampler (`_mlx_greedy_sample`) only returns sampled token ids; it never computes log-softmax. The upstream OpenAI front-end (shared code) assumes the engine populates logprobs aligned 1:1 with token ids and crashes when it doesn't. This is a Metal-port completeness gap, distinct from the upstream prefix-caching + logprobs interaction tracked under [vllm #5344](https://github.com/vllm-project/vllm/issues/5344) / [#5890](https://github.com/vllm-project/vllm/issues/5890) / [#8268](https://github.com/vllm-project/vllm/issues/8268). Disabling prefix caching (`make serve PREFIX_CACHING=off`) does **not** fix it — confirmed by direct curl probe.

Because `lm-evaluation-harness` scores multiple-choice tasks by computing log-likelihoods of candidate answers, `make eval` cannot run end-to-end against the metal server. Two workarounds:

1. **Generation-based MC (reported path):** `make eval-gen` uses `eval_runner/gen_mc_eval.py` to render the question with `A/B/C/D` choices, generate the answer letter, and exact-match against gold. Runs on the fast metal server. Covers `custom_commonsense` and `mmlu`. HellaSwag's full-sentence endings don't fit a clean letter-answer reformulation, so it's omitted from this path. For instruct-tuned models like `Qwen2.5-3B-Instruct`, generation-based MC accuracy is typically within 1–3 points of loglikelihood scoring and preserves model rankings, but absolute numbers are not directly comparable to published loglikelihood-based MMLU/HellaSwag scores. **This is the metric we report.**
2. **CPU server (canonical metric, experimental):** `make serve-cpu` runs upstream vLLM on CPU (the metal plugin is skipped via `VLLM_PLUGINS=""`). Logprobs work end-to-end — verified by a direct probe with `echo=True, logprobs=1`, which returns the prompt-aligned logprob list cleanly. Trade-off: CPU inference is ~40 tok/s prompt-eval on Apple Silicon, so a full benchmark would take hours, and under sustained eval traffic we observed intermittent `Connection refused` errors that we couldn't fully resolve in the available time. The wiring is preserved (`serve.py --backend cpu`, `make serve-cpu`) for future debugging or for users on a less-loaded machine, but no canonical loglikelihood numbers are produced in this run.

Net: every reported accuracy number in this repo comes from `make eval-gen` against the metal server.

## Part C: Performance & Scaling

Load generator that sweeps across concurrency levels with short and long prompts. Collects TTFT, tokens/sec, and P50/P95/P99 latency metrics.

```bash
make perf
jupyter notebook perf/analysis.ipynb
```

## Part D: Guardrails & Determinism

Validates deterministic inference (fixed seeds, temperature=0), output schema conformance, and edge cases.

```bash
make guardrails
```

## Part E: Benchmark Improvement

Inference-time optimization targeting ARC-Challenge with prompt rewriting, few-shot retrieval, chain-of-thought, and self-consistency voting.

```bash
make improve
```

## Results Summary

All numbers below come from this run, against `Qwen/Qwen2.5-3B-Instruct` served via `vllm-metal 0.2.0` on Apple Silicon. Raw artifacts live under `eval_runner/results/`, `perf/metrics.csv`, `guardrails/run_output.txt`, and `improve/results/`.

### Part B — Evaluation (generation-based MC)

| Task | Strategy | Accuracy | n_samples | n_parseable | n_errors | Wall clock |
|---|---|---:|---:|---:|---:|---:|
| `mmlu` (full) | gen-eval, exact-match-letter | **0.6556** | 14,042 / 14,042 | 99.99% | 0 | 34 min |
| `hellaswag` | — | n/a | — | — | — | — |
| `custom_commonsense` | gen-eval, exact-match-letter | **1.000** | 50 / 50 | 100% | 0 | <1 min |

- MMLU is the **full 14,042-sample test** (no LIMIT). Accuracy 65.56%, 0 errors, 6.8 req/s sustained on the metal server. The number sits ~0pp above Qwen2.5-3B-Instruct's published MMLU (~65.6%), validating that the generation-based MC path tracks the canonical loglikelihood metric closely on this benchmark — at least for instruct-tuned models that emit single-letter answers reliably.
- HellaSwag is omitted from the generation-based path because its choices are full-sentence completions, not letter answers — no clean reformulation exists. Loglikelihood scoring would be required and is blocked by the `vllm-metal` 0.2.0 logprobs bug (see note above). The CPU server profile (`make serve-cpu`) does compute logprobs cleanly but proved too slow / connection-flaky for a full HellaSwag run; this is the one explicit-brief gap in the submission.
- `custom_commonsense` is a 50-question hand-built benchmark (`eval_runner/custom_task/`); 100% accuracy reflects the questions being well within the model's competence rather than benchmark difficulty.

### Part C — Performance & Scaling

Load test sweep across concurrency ∈ {1, 2, 4, 8, 16, 32}, prompt length ∈ {short, long}, stop conditions ∈ {none, single_newline, double_newline}. 1,134 records in `perf/metrics.csv`. Headline numbers (concurrency=1, short prompt, no stop sequence):

- **TTFT P50: 244 ms · TTFT P95: 2,738 ms · Tokens/sec: 24.1**

At concurrency=16 with double-newline stops the server sustains ~270 tok/s aggregate. Throughput collapses past concurrency=32 due to KV-cache pressure on the metal backend. Plots in `perf/analysis.ipynb`.

### Part D — Guardrails & Determinism

`make guardrails` → **7/7 PASS** (`guardrails/run_output.txt`):

1. Deterministic generation (temperature=0, seed=42) — 3 identical responses
2. Determinism across varied prompts — 3/3 prompts yielded deterministic outputs
3. MCQ output schema (`^[A-D]`) — 5/5 matched
4. JSON output schema — parsed cleanly, all required fields present
5. Edge case: empty prompt — server returned HTTP 400 (correct)
6. Edge case: ~3,000-token prompt — handled, returned 20-token completion
7. Edge case: prompt-injection attempt — model responded normally, leaked prompt-injection token but did not exfiltrate or change behavior

### Part E — Benchmark Improvement (ARC-Challenge)

**Full test (*n*=1,172, greedy decoding)** — primary result (`improve/results/results_baseline.json`, `results_few_shot.json`):

| Strategy | Accuracy | 95% CI | Δ vs baseline | McNemar *p* (vs baseline) |
|---|---:|---|---:|---:|
| Baseline (zero-shot) | **81.74%** | [79.78, 83.79] | — | — |
| + Few-shot (k=3 train demos, FAISS cosine) | **81.57%** | [79.35, 83.79] | **−0.17 pp** | **0.91** |

**Headline:** few-shot ICL is **not significantly different** from zero-shot on the full ARC test; the +2.5 pp assignment target is **not met**.

**Same 500 question IDs:** baseline subset **82.40%**, few-shot greedy **83.40%**, few-shot + self-consistency (*k*=3) **83.40%** (417/500 each for greedy vs SC). SC did **not** move the aggregate versus greedy few-shot; McNemar greedy vs SC *p*=0.62 (*b*=2, *c*=2).

Earlier `LIMIT=100` ablations were exploratory only and are summarized in `improve/report.md`; the committed JSONs focus on the full-test / same-500-ID results.

## Final Summary — Best Improvement and Lessons Learned

The headline quantitative result is the **full ARC-Challenge test (*n*=1,172)**: baseline **81.74%**, FAISS few-shot **81.57%**, McNemar *p*=0.91 — **no lift**. On the **same 500 questions**, few-shot + SC (*k*=3) matched greedy few-shot (**83.40%**, 417/500). The earlier *n*=100 slice had suggested +1.0 pp for few-shot; **full-test evaluation overrides that.**

For the short narrative version (submission checklist item), see [`FINAL_SUMMARY.md`](FINAL_SUMMARY.md). Lessons that survived contact with the full test:

1. **Infrastructure vs prompts.** When `vllm-metal` breaks logprobs, you either fix the server stack or adopt a parallel scoring path (`gen_mc_eval.py`) — engineering dominates headline metrics.
2. **Slice ≠ benchmark.** Do not ship quantitative claims from a prefix slice when the full test fits on disk.
3. **Long jobs need power.** Battery drain / sleep stops everything mid-request — plug in for full benchmark runs.

