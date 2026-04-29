LLM Systems & Evaluation Interview (Time: 4 hours)
Context
In this assignment, you will architect and refine a scaled-down version of an internal LLM evaluation pipeline.
You’ll use two real-world components:
vLLM: a high-throughput inference engine powering production-grade deployments
lm-evaluation-harness: the industry-standard benchmark framework by EleutherAI


Part A: Serving
Spin up a vLLM inference server for any open-weight model (e.g., Llama 3, Mistral, Phi). Expose a simple REST or OpenAI-compatible /generate endpoint.
Requirements:
Enable continuous batching and paged attention (default in vLLM).
Write a Python client that supports:
Streaming token generation
Configurable decoding parameters (max_tokens, temperature, top_p, stop sequences)
Validate that multiple clients can query concurrently without performance degradation.


Deliverables:
serve/ folder with serve.py and client.py
One-line startup command (make serve or python serve.py)
Sample script that runs a few prompt generations
Part B: Evaluation
Integrate your served model with the lm-evaluation-harness, so it can run standardized benchmarks just like a hosted model API.
Tasks:
Write a custom model wrapper that queries your vLLM endpoint.
Evaluate on:
Two official tasks (e.g., MMLU and HellaSwag)
One small, custom JSON-based benchmark you design yourself
Add caching for repeated prompts so runs are deterministic and efficient.


Deliverables:
eval_runner/ folder with:
vllm_model.py (wrapper)
run_eval.py (runner script)
results/ with benchmark outputs and a summary table
Part C: Performance & Scaling
Tasks:
Implement a load generator that sends concurrent requests (short vs long prompts).
Collect and log:
Time-to-first-token (TTFT)
Tokens per second (TPOT)
P50 / P95 / P99 latency
GPU utilization (if available)
Compare across batch sizes, caching, and stop-sequence settings.


Deliverables:
perf/ folder with:
load_test.py
metrics.csv
analysis.ipynb with plots and short commentary


Part D: Guardrails & Determinism
Add basic reliability guardrails:
Implement deterministic mode (set seeds, temperature=0, top_p=1).
Verify identical prompts yield identical responses.
Add lightweight validation logic (regex or schema) for your custom task outputs.


Deliverables:
guardrails/validate.py
Short README describing what you tested and where nondeterminism persists
Part E: Benchmark Improvement (No Finetuning)
Choose one benchmark — HellaSwag, MMLU, or ARC-Challenge — and improve its score using only inference-time optimization.
Constraints:
You must use the same model and vLLM configuration.
No finetuning or parameter updates.
Changes must be reproducible and statistically valid (p < 0.05).


Allowed improvement levers:
Prompt optimization:
Template rewriting and instruction design
Automatic few-shot selection (semantic similarity or clustering)
Chain-of-thought or rationale-augmented prompts
Self-consistency (k-sample decoding + majority voting)
Prompt ensembling across phrasing variants


Decoding optimization:
Temperature, top-p, top-k tuning
Stop-sequence refinement
Output normalization or regex-based mapping


Retrieval augmentation:
Deterministic retrieval from a local, static corpus


Confidence calibration:
Filtering or rescoring based on logprobs or entropy


Deliverables:
improve/
 ├── prepare_data.py
 ├── optimize_prompt.py
 ├── infer.py
 ├── eval.sh
 └── report.md
Your report.md (400–700 words) should include:
Baseline vs improved results (with 95% confidence intervals)
Ablation study showing the impact of each change
10+ before/after examples with short analysis
Cost and latency trade-offs
Exact seeds, decoding settings, and configurations


Target lifts:
HellaSwag: +3.0 accuracy
MMLU (subject group): +2.0 accuracy
ARC-Challenge: +2.5 accuracy
Submission
Email a GitHub repository or zip file to mle-interviewers@mercor.com.
This should contain:
All code folders (serve/, eval_runner/, perf/, guardrails/, improve/)
Makefile and README.md
results/ and metrics.csv
A short final summary: the story of your best improvement and what you learned
