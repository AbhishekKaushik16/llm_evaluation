# ARC-Challenge Benchmark Improvement Report

## Objective

Improve ARC-Challenge accuracy by at least +2.5 points using only inference-time optimization (no finetuning or parameter updates), with the same model and vLLM configuration.

## Model & Configuration

- **Model**: `Qwen/Qwen2.5-3B-Instruct`
- **Serving**: vLLM (`vllm-metal` 0.2.0) with continuous batching, paged attention, prefix caching, seed=42
- **Hardware**: Apple Silicon (MacBook, M-series)
- **Run scope**: Full ARC-Challenge **test** split (1,172 questions) for baseline vs FAISS few-shot (greedy). Earlier exploratory runs used `LIMIT=100` (wide CIs) and are summarized below without keeping all intermediate JSON files.

## Baseline vs Improved Results (full test, n=1,172)

Numbers from `improve/results/results_baseline.json` and `improve/results/results_few_shot.json`. Confidence intervals are bootstrap (n=1000 resamples). Paired comparison uses McNemar’s test on per-question correctness.

| Strategy | Accuracy | 95% CI | Δ vs baseline | McNemar *p* |
|----------|---------:|--------|--------------:|------------:|
| Baseline (zero-shot) | **81.74%** | [79.78%, 83.79%] | — | — |
| + Few-shot (k=3 train demos, FAISS cosine) | **81.57%** | [79.35%, 83.79%] | **−0.17 pp** | **0.91** |

McNemar contingency: 38 questions where baseline wrong & few-shot right; 40 where baseline right & few-shot wrong (*b*=38, *c*=40). **No significant lift** — few-shot is essentially tied with zero-shot on the full test.

**None of these approaches reached the assignment’s +2.5 pp ARC target.** On the full test the headline story changes versus the small-*n* slice: the first-100-question window had looked like +1.0 pp for few-shot; at *n*=1,172 that signal disappears into noise.

### Earlier ablation (n=100, greedy)

The first-pass ablation covered instruction prompting, CoT, combined prompting, and combined+SC on a 100-question cap. Those rows are **exploratory only** (±~8 pp CIs), so the committed results focus on the full-test baseline/few-shot comparison and the same-500-ID SC comparison.

### Merged 500-question subset — few-shot vs few-shot + SC (*k*=3)

Completed (`results_few_shot_sc3.json`, ~36 min). On the **same 500 IDs** as this run:

| Strategy | Accuracy | Correct |
|---|---:|---:|
| Baseline (subset) | 82.40% | 412 / 500 |
| Few-shot greedy | **83.40%** | 417 / 500 |
| Few-shot + SC (*k*=3, *T*=0.7) | **83.40%** | 417 / 500 |

McNemar baseline vs SC *p*=0.52; greedy few-shot vs SC *p*=0.62 (*b*=2, *c*=2). **Self-consistency did not change aggregate accuracy** versus greedy few-shot — identical 417 / 500.

## Honest Take

- **Full test > slice.** A convenient prefix of the ARC test set is not a surrogate for the full benchmark — ordering by grade makes early questions easier; small *n* inflates CI width.
- **Few-shot here did not beat baseline at scale.** Retrieval-augmented ICL is still the most principled lever we tried, but on Qwen2.5-3B-Instruct at greedy decoding it neither helps nor hurts meaningfully on the full 1,172-question test.
- **Self-consistency completed on 500 prompts** and matched greedy few-shot exactly at **417 / 500** — variance reduction without net lift (paired disagreements canceled).

## Techniques Applied

### 1. Prompt Template Rewriting
Replaced the minimal default prompt with a structured instruction telling the model it is an expert at science questions and should select the single best answer. Provides task framing.

### 2. Automatic Few-Shot Selection
Embedded all 1,119 ARC-Challenge train questions using `all-MiniLM-L6-v2` (sentence-transformer, 384-dim) and built a FAISS inner-product index over L2-normalized vectors (cosine similarity). For each test question, retrieve the top-3 most semantically similar train examples as few-shot demonstrations. No manual example curation.

### 3. Chain-of-Thought Prompting
Added "Let's think step by step" elicitation. Few-shot examples include a brief reasoning prefix demonstration. Encourages multi-step reasoning over pattern-matching.

### 4. Self-Consistency (k=5 Majority Voting)
For the combined strategy, sample k=5 responses with temperature=0.7 and take the majority vote per question. Reduces variance from stochastic decoding. Wall-clock cost was ~13 minutes for n=100 (5× slower than greedy combined), as expected. Final accuracy: 80.0% — identical to greedy combined and identical to baseline. SC successfully reduced sampling-variance noise but produced no underlying lift to recover, confirming that the techniques don't disagree productively on this slice.

### 5. Output Normalization
Regex-based answer extraction handling: direct letter, "The answer is X", letter with punctuation, fallback to last letter mention. Catches the model's intended answer despite verbose outputs.

## Cost and Latency Trade-offs

| Strategy | Avg Output Tokens | Wall Clock (n=100, conc=8) | Relative Cost |
|----------|------------------:|---------------------------:|--------------:|
| Baseline | ~5 | ~3.5 min | 1.0× |
| Instruction | ~5 | ~3.5 min | 1.0× |
| Few-shot (k=3) | ~10 | ~3.5 min | ~1.5× (longer prompts) |
| CoT | ~100 | ~3.5 min | ~3.0× |
| Combined (greedy) | ~100 | ~3.5 min | ~3.0× |
| Combined + SC (k=5) | ~100 × 5 | ~13 min (measured) | ~15× |

Self-consistency is significantly more expensive but typically the largest variance reducer. Worth running on a full benchmark; not worth running on this 100-question slice given the sampling noise.

## Reproducibility

- **Seed**: 42 (vLLM server, embedding deterministic, greedy decoding)
- **Embedding model**: `all-MiniLM-L6-v2`
- **Self-consistency**: temperature=0.7, top_p=0.95, k=5, fixed seed sequence per question
- **Greedy strategies**: temperature=0.0, top_p=1.0
- **FAISS index**: cosine similarity via inner product on L2-normalized vectors
- **Run command (full test, greedy baseline + few-shot):** run `python improve/infer.py --strategy baseline` then `python improve/infer.py --strategy few_shot` with server up. For the 500-question SC leg, run `python improve/infer.py --strategy few_shot --self-consistency-k 3 --sc-temperature 0.7 --limit 500`.

## Statistical Validity

- Bootstrap CIs (n=1000 resamples) reported per strategy on the full test.
- McNemar’s test for paired baseline vs few-shot on *n*=1,172: *p*=0.91 — **no significant difference**.
- The exploratory *n*=100 ablations remain useful for qualitative comparisons (instruction vs CoT vs combined) but should not drive quantitative claims.

## Before/After Examples

See `improve/results/before_after_examples.md` for question-level diffs from the earlier pipeline run (`LIMIT=100`). Those comparisons pair baseline vs **combined**, not few-shot; regenerate after choosing a primary “improved” strategy if needed.

## Final Summary — The Story and What I Learned

**What held up on the full ARC-Challenge test (*n*=1,172).** Zero-shot baseline landed at **81.74%** [79.78, 83.79]. FAISS-retrieved few-shot (three train demos selected by cosine similarity over `all-MiniLM-L6-v2` embeddings) landed at **81.57%** [79.35, 83.79] — a **−0.17 pp** delta with McNemar *p*=0.91. So the small-*n* slice had overstated a few-shot gain; at scale, **few-shot does not beat baseline** under greedy decoding.

**What I learned.**

1. **Infrastructure still dominates wall-clock.** The `vllm-metal` logprobs gap forced a generation-based MC path for lm-eval; that remains the central engineering constraint.
2. **Pick *n* before trusting deltas.** A 100-question prefix is not the full benchmark — ordering effects and CI width make it misleading.
3. **Battery / sleep kills long jobs.** Plug in power for multi-hour evals; `infer.py` retries HTTP failures and isolates per-item errors.

**Net.** The definitive headline result is the **full-test tie** between baseline and FAISS few-shot (*n*=1,172). The **merged 500-ID subset** shows few-shot greedy **83.40%** vs few-shot + SC (*k*=3) **83.40%** (same 417 / 500) — SC added **no** aggregate lift. The +2.5 pp ARC assignment target is **not met**.
