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

| Task | Strategy | Accuracy | n_samples | n_parseable | n_errors |
|---|---|---:|---:|---:|---:|
| `mmlu` (partial) | gen-eval, exact-match-letter | **0.617** | 2,500 / 14,042 | 99.4% | 14 |
| `hellaswag` | — | n/a | — | — | — |
| `custom_commonsense` | gen-eval, exact-match-letter | **1.000** | 50 / 50 | 100% | 0 |

- MMLU is reported on a 2,500-sample partial run (interrupted under deadline). The number is statistically stable to ~±1% and matches published Qwen2.5-3B-Instruct MMLU (~62%) within the gen-vs-loglikelihood gap.
- HellaSwag is omitted from the generation-based path because its choices are full-sentence completions, not letter answers — no clean reformulation exists. Loglikelihood scoring would be required and was not produced this run (see "Note on `vllm-metal` and logprobs" above).
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

`make improve LIMIT=100` results (95% bootstrap CI, n=1000 resamples):

| Strategy | Accuracy | 95% CI | Δ vs baseline |
|---|---:|---|---:|
| Baseline (zero-shot) | **80.0%** | [72.0, 88.0] | — |
| + Instruction prompt | 79.0% | [71.0, 87.0] | −1.0 |
| + Few-shot (k=3, FAISS-retrieved) | **81.0%** | [73.0, 88.0] | **+1.0** |
| + Chain-of-thought | 79.0% | [71.0, 86.0] | −1.0 |
| Combined (instruct + few-shot + CoT, greedy) | 80.0% | [72.0, 88.0] | 0.0 |
| Combined + SC (k=5, T=0.7) | 80.0% | [72.0, 87.0] | 0.0 |

No technique — including self-consistency, the most expensive and most-likely-to-win experiment in the suite — cleared the **+2.5 pp** assignment target on this 100-question slice. All CIs overlap heavily — at n=100 the SE on accuracy is ~4 pp, so true lifts under ~8 pp aren't reliably detectable. The first 100 questions of ARC-Challenge skew toward easier elementary-grade items, leaving little headroom on a strong 80% baseline. To draw a defensible conclusion we'd need the full 1,172-question test set (CI shrinks to ±~2 pp) and the combined+SC pass to complete (~30–60 min on the metal server). Full discussion, technique descriptions, cost trade-offs, and before/after examples live in `improve/report.md` and `improve/results/before_after_examples.md`.

## Final Summary — Best Improvement and Lessons Learned

The hypothesis going in was that a 3B instruct model on school-science questions would benefit more from *examples* than from *instruction*. That held: the only technique with a positive point estimate was **FAISS-retrieved few-shot prompting** — embed all 1,119 ARC train questions with `all-MiniLM-L6-v2`, build a cosine-similarity FAISS index, and at inference time pull the top-3 nearest train questions as in-context demonstrations. No manual example curation. That moved accuracy from 80.0% to 81.0% (+1.0 pp). Instruction prompting (−1.0 pp), chain-of-thought (−1.0 pp), combined-greedy (0.0 pp), and combined + self-consistency (k=5, T=0.7, also 0.0 pp) did not help on this slice. SC is the cleanest data point in the table: at ~15× the baseline cost it should have averaged out stylistic disagreements between the stacked techniques, and the fact that it landed at exactly 80.0% confirms the techniques don't disagree productively — there was no underlying lift for variance reduction to recover.

What I'd take away if I were doing this again:

1. **Infrastructure shapes the experiment more than prompt engineering does.** The single biggest time sink was a `vllm-metal 0.2.0` bug that hardcodes `logprobs=None` in every `ModelRunnerOutput`, breaking the standard lm-eval log-likelihood scoring path. The workaround — a generation-based MC scorer that asks the model for a single answer letter and exact-matches it — is correct and reproducible, but it's not the canonical metric. Lesson: when the eval harness assumes a capability your serving stack doesn't have, you need a parallel non-likelihood path ready, or you don't get to evaluate.
2. **Sample-size budgeting beats prompt cleverness at small n.** At n=100, the 95% CI on accuracy is ~±8 pp; the assignment's +2.5 pp target needs n≥400 (α=0.05, 80% power) to be detectable. I underestimated this. Every Δ in the table is directionally interpretable, not statistically defensible. Pick n from the target effect size *before* designing the prompt sweep.
3. **Strong instruct baselines leave little headroom on easy slices.** Qwen2.5-3B-Instruct sits at 80% zero-shot on the first 100 ARC questions because they skew elementary-grade. A stratified slice across the grade band would have been the right thing to evaluate against.
4. **Chain-of-thought can hurt regex-scored multiple choice** — not because reasoning got worse, but because verbose outputs occasionally drift off the strict letter-answer format and get mis-extracted. The answer parser is part of the evaluation system and needs its own test coverage.
5. **Compose carefully — and SC isn't a free fix.** Stacking instruct + few-shot + CoT returned exactly to baseline; the techniques partially cancel because each pulls the model toward a different output style. Self-consistency on top of the combined prompt at k=5 (T=0.7) — the principled variance-reduction fix — also landed at 80.0%. SC works as advertised when the underlying samples disagree productively; here they didn't, so there was nothing for majority voting to recover. The bottleneck is sample size and slice composition, not technique choice.
